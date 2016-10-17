//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <functional>
#include "Basics.h"
#include "MPIWrapper.h"
#include "CNTKLibrary.h"
#include "DistributedCommunicator.h"
#include "CUDAPageLockedMemAllocator.h"
#include "MatrixQuantizerImpl.h"
#include "GPUDataTransferer.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    static Microsoft::MSR::CNTK::MPIWrapperPtr s_mpi;

    DistributedCommunicatorPtr MPICommunicator()
    {
        return std::make_shared<MPICommunicatorImpl>();
    }

    MPICommunicatorImpl::Buffer MPICommunicatorImpl::AllocateIntermediateBuffer(int deviceID, size_t totalSize)
    {
        assert(deviceID >= 0);
        Buffer buffer;
        buffer.totalSize = totalSize;
        buffer.data = std::shared_ptr<void>(
            CUDAPageLockedMemAllocator::Malloc(totalSize, deviceID),
            [deviceID](void* p) { CUDAPageLockedMemAllocator::Free(p, deviceID); });
        return buffer;
    }

    inline size_t GetBufferSize(const NDArrayViewPtr& viewPtr)
    {
        return viewPtr->Shape().TotalSize() * DataTypeSize(viewPtr->GetDataType());
    }

    inline void* GetDataBuffer(const NDArrayViewPtr& viewPtr)
    {
        if (viewPtr->GetDataType() == DataType::Float)
            return const_cast<float*>(viewPtr->DataBuffer<float>());
        if (viewPtr->GetDataType() == DataType::Double)
            return const_cast<double*>(viewPtr->DataBuffer<double>());
        LogicError("Unknown DataType");
        return nullptr; // Make compiler happy.
    }

    inline DeviceDescriptor GetNonCPUDevice(const std::vector<ValuePtr>& values)
    {
        auto device = std::find_if(values.begin(), values.end(), [](const ValuePtr v) { return v->Device().Type() != DeviceKind::CPU; });
        return values.end() == device ? DeviceDescriptor::CPUDevice() : (*device)->Device();
    }

    MPICommunicatorImpl::MPICommunicatorImpl()
    {
        if (!s_mpi)
            s_mpi = MPIWrapper::GetInstance(true);

        m_mpi = s_mpi;
        m_currentWorker.m_globalRank = s_mpi->CurrentNodeRank();
        m_currentWorker.m_hostId = std::wstring(s_mpi->CurrentNodeName());
        for (size_t i = 0; i < s_mpi->NumNodesInUse(); ++i)
        {
            if (i == m_currentWorker.m_globalRank)
                m_workers.insert(m_currentWorker);
            else
                m_workers.insert({ i,  L"" });
        }
    }

    std::vector<int> MPICommunicatorImpl::Initialize(const std::vector<ValuePtr>& values)
    {
        assert(CPUDEVICE < 0); // just in case somebody decides to change CPUDEVICE macro.

        std::vector<int> indices(values.size(), CPUDEVICE);
        int numGPUValues = 0;
        DeviceDescriptor lastGpuDevice = DeviceDescriptor::CPUDevice();
        for (auto i = 0; i < values.size(); ++i)
        {
            auto& value = values[i];
            auto view = value->Data();
            auto device = view->Device();

            // Make sure none of the values are sparse - we currently do not support aggregation of sparse matrices
            if (view->GetStorageFormat() != StorageFormat::Dense)
                RuntimeError("Aggregation for sparse matrices is currently not supported!");

            if (device.Type() != DeviceKind::GPU)
                continue;

            if (lastGpuDevice.Type() == DeviceKind::CPU)
                lastGpuDevice = device;
            else if (device.Id() != lastGpuDevice.Id()) // For the time being, assume all devices have the same id.
                LogicError("Not all values share the same GPU device id");

            auto index = numGPUValues++;
            if (m_gpuDataTransferers.size() < numGPUValues)
                m_gpuDataTransferers.push_back(std::make_unique<GPUDataTransferer>(device.Id(), true));

            if (m_intermediateCPUBuffers.size() < numGPUValues)
                m_intermediateCPUBuffers.push_back(Buffer());

            auto requiredSize = GetBufferSize(view);
            if (m_intermediateCPUBuffers[index].totalSize < requiredSize)
                m_intermediateCPUBuffers[index] = AllocateIntermediateBuffer(device.Id(), requiredSize);

            indices[i] = index;
        }
        return indices;
    }

    std::unordered_set<DistributedWorkerDescriptor> MPICommunicatorImpl::Workers() const
    {
        return m_workers;
    }

    const DistributedWorkerDescriptor& MPICommunicatorImpl::CurrentWorker() const
    {
        return m_currentWorker;
    }

    std::vector<ValuePtr> MPICommunicatorImpl::Aggregate(const std::vector<ValuePtr>& values,
        const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        std::vector<ValuePtr> outputValues;
        for (const auto& inputValue : values)
        {
            const auto inputView = inputValue->Data();
            const auto& outputView = MakeSharedObject<NDArrayView>(inputView->GetDataType(), inputView->Shape(), inputView->Device());
            const auto& inputMask = inputValue->Mask();
            const auto& outputMask = MakeSharedObject<NDMask>(inputMask->Shape(), inputMask->Device());
            outputValues.push_back(MakeSharedObject<Value>(outputView, outputMask));
        }

        auto device = GetNonCPUDevice(values);
        if (device.Type() != DeviceKind::CPU)
        {
            // Since we will be copying the gradients asynchronously, let us
            // ensure that the gradient matrices have been computed before starting to aggregate
            // them asynchronously on another thread. This essentially means that when we are using
            // a GPU device, we will synchronize on the main GPU compute stream before starting
            // the gradient aggregation asynchronously on a separate stream
            std::unique_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent(MatrixComputeStreamEvent::Create(device.Id()));
            mainStreamSyncEvent->SynchronizeDataTransferFetchStreamWithEvent<float>();
        }

        AggregateImpl(values, outputValues, sendToWorkers);
        return outputValues;
    }

    DistributedCommunicatorPtr MPICommunicatorImpl::SubGroup(const std::unordered_set<DistributedWorkerDescriptor>&) const
    {
        NOT_IMPLEMENTED;
    }

    std::unordered_set<ValuePtr> MPICommunicatorImpl::Concatenate(const std::unordered_set<ValuePtr>&, const std::unordered_set<DistributedWorkerDescriptor>&)
    {
        NOT_IMPLEMENTED;
    }

    std::future<std::vector<ValuePtr>> MPICommunicatorImpl::AggregateAsync(const std::vector<ValuePtr>& values,
        const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        auto device = GetNonCPUDevice(values);

        std::shared_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent;
        if (device.Type() != DeviceKind::CPU)
            mainStreamSyncEvent.reset(MatrixComputeStreamEvent::Create(device.Id()));

        return std::async(std::launch::async, [this, &values, &sendToWorkers, device, mainStreamSyncEvent]()
        {
            if (device.Type() != DeviceKind::CPU)
            {
                // We are starting on a new thread. Make sure the new thread is setup to use the right device
                // TODO: SetDevice is type agnostic, move it to the base matrix class. 
                Matrix<float>::SetDevice(device.Id());

                // Since we will be copying the gradients asynchronously, let us
                // ensure that the gradient matrices have been computed before starting to aggregate
                // them asynchronously on another thread. This essentially means that when we are using
                // a GPU device, we will synchronize on the main GPU compute stream before starting
                // the gradient aggregation asynchronously on a separate stream
                mainStreamSyncEvent->SynchronizeDataTransferFetchStreamWithEvent<float>();
            }

            return this->Aggregate(values, sendToWorkers);
        });
    }

    void MPICommunicatorImpl::AggregateInPlace(const std::vector<ValuePtr>& values,
        const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        auto device = GetNonCPUDevice(values);
        if (device.Type() != DeviceKind::CPU)
        {
            // Since we will be copying the gradients asynchronously, let us
            // ensure that the gradient matrices have been computed before starting to aggregate
            // them asynchronously on another thread. This essentially means that when we are using
            // a GPU device, we will synchronize on the main GPU compute stream before starting
            // the gradient aggregation asynchronously on a separate stream
            std::unique_ptr<MatrixComputeStreamEvent> mainStreamSyncEvent(MatrixComputeStreamEvent::Create(device.Id()));
            mainStreamSyncEvent->SynchronizeDataTransferFetchStreamWithEvent<float>();
        }
        AggregateImpl(values, values, sendToWorkers);
    }

    void  MPICommunicatorImpl::AggregateImpl(const std::vector<ValuePtr>& inputValues,
        const std::vector<ValuePtr>& outputValues,
        const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers)
    {
        if (m_mpi->NumNodesInUse() == 1) // No need to aggregate anything.
            return;

        UNUSED(sendToWorkers);
        assert(inputValues.size() == outputValues.size());

        auto numValues = inputValues.size();
        if (numValues == 0)
        {
            return;
        }

        auto prepared = Initialize(inputValues);
        auto gpuIndexVector = prepared;

        // for all values residing on GPU initiate async transfer to CPU buffers.
        for (auto i = 0; i < numValues; ++i)
        {
            auto& value = inputValues[i];
            auto view = value->Data();
            auto gpuIndex = gpuIndexVector[i];
            if (gpuIndex != CPUDEVICE)
            {
                auto& transferer = m_gpuDataTransferers[gpuIndex];
                auto& buffer = m_intermediateCPUBuffers[gpuIndex];
                transferer->CopyGPUToCPUAsync(GetDataBuffer(view), GetBufferSize(view), buffer.data.get());
            }
        }

        std::vector<MPI_Request> allReduceRequests(numValues);
        for (auto i = 0; i < numValues; ++i)
        {
            auto gpuIndex = gpuIndexVector[i];
            if (gpuIndex != CPUDEVICE)
            {
                // TODO: actually, we can start reducing all cpu values first, and then wait for the gpu->cpu transfer to finish.
                m_gpuDataTransferers[gpuIndex]->WaitForCopyGPUToCPUAsync();
            }

            auto& inputValue = inputValues[i];
            auto inputView = inputValue->Data();
            auto numElements = inputView->Shape().TotalSize();
            auto dataType = inputView->GetDataType();

            auto& outputValue = outputValues[i];
            auto outputView = outputValue->Data();

            assert(numElements == outputView->Shape().TotalSize());
            assert(dataType == outputView->GetDataType());
            assert(inputView->Device() == outputView->Device());

            void* inputData = (gpuIndex != CPUDEVICE) ? m_intermediateCPUBuffers[gpuIndex].data.get() : GetDataBuffer(inputView);
            void* outputData = (gpuIndex != CPUDEVICE) ? m_intermediateCPUBuffers[gpuIndex].data.get() : GetDataBuffer(outputView);

            if (dataType == DataType::Float)
            {
                if (inputData == outputData)
                    m_mpi->AllReduceAsync<float>(static_cast<float*>(outputData), numElements, &allReduceRequests[i]);
                else
                    m_mpi->AllReduceAsync<float>(static_cast<float*>(inputData), static_cast<float*>(outputData), numElements, &allReduceRequests[i]);
            }
            else if (dataType == DataType::Double)
            {
                if (inputData == outputData)
                    m_mpi->AllReduceAsync<double>(static_cast<double*>(outputData), numElements, &allReduceRequests[i]);
                else
                    m_mpi->AllReduceAsync<double>(static_cast<double*>(inputData), static_cast<double*>(outputData), numElements, &allReduceRequests[i]);
            }
            else
                LogicError("Unknown DataType");
        }

        // wait for async all reduce to complete. As soon as one of the requests is finished,
        // check if corresponding value is gpu bound and, if it is the case, initiate a cpu-to-gpu transfer.
        size_t numAllReduceRequestsCompleted = 0;
        while (numAllReduceRequestsCompleted < numValues)
        {
            int idx = MPI_UNDEFINED;
            m_mpi->WaitAny(allReduceRequests.data(), (int)allReduceRequests.size(), &idx);
            if (idx == MPI_UNDEFINED)
            {
                break;
            }

            numAllReduceRequestsCompleted++;

            auto gpuIndex = gpuIndexVector[idx];

            if (gpuIndex != CPUDEVICE)
            {
                auto view = outputValues[idx]->Data();
                auto size = GetBufferSize(view);
                auto& transferer = m_gpuDataTransferers[gpuIndex];
                auto& buffer = m_intermediateCPUBuffers[gpuIndex];
                transferer->CopyCPUToGPUAsync(buffer.data.get(), size, GetDataBuffer(view));
            }
        }

        // TODO: Should not wait, simply publishing event on the compute stream should be sufficient.
        for (auto i = 0; i < numValues; ++i)
        {
            auto gpuIndex = gpuIndexVector[i];
            if (gpuIndex != CPUDEVICE)
                m_gpuDataTransferers[gpuIndex]->WaitForCopyCPUToGPUAsync();
        }
    }

    void MPICommunicatorImpl::QuantizedAggregate(const std::vector<ValuePtr>& /*inValues*/,
        const std::unordered_set<ValuePtr>& /*inPreviousQuantizationResidues*/,
        const std::unordered_set<DistributedWorkerDescriptor>& /*sendToWorkers*/,
        const std::unordered_set<ValuePtr>& /*aggregatedOutputs*/,
        const std::unordered_set<ValuePtr>& /*newQuantizationResidues*/)
    {
        NOT_IMPLEMENTED;
    }
}