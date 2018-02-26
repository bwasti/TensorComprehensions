/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tc/core/opencl/opencl_tc_executor.h"
#include "tc/core/polyhedral/mapped_scop.h"

namespace tc {

namespace {

// Append ordered values to the kernel name, separated by "_".
std::string specializeKernelName(
    const std::string& kernelName,
    std::vector<int> params) {
  std::stringstream ss;
  ss << kernelName;
  for (auto i : params) {
    ss << "_" << i;
  }
  return ss.str();
}

std::vector<int> narrowParamsVector(const std::vector<long>& params) {
  std::vector<int> result;
  result.reserve(params.size());
  for (auto l : params) {
    CHECK_GE(l, std::numeric_limits<int>::min()) << "parameter underflows int";
    CHECK_LE(l, std::numeric_limits<int>::max()) << "parameter overflows int";
    result.push_back(static_cast<int>(l));
  }
  return result;
}
} // namespace

void OpenCLTcExecutor::compile(const tc::MappingOptions& options) {
  if (rtcFun) {
    throw std::runtime_error{
        "OpenCLTcExecutor::compile cannot be called multiple tines."};
  }
  execInfo_.options = options.toProtobufSerializedString();
  compileWithTcMapper();
  rtcFun = OpenCLRTCFunction::Compile(kernelSpecializedName, openclSource);
}

void OpenCLTcExecutor::compileWithTcMapper() {
  auto scopTmp = polyhedral::Scop::makeScop(ctx_, halideComponents_);
  auto globalParameterContext =
      scopTmp->makeContextFromInputs(extractRawPtrs(execInfo_.inputsInfo));
  scopTmp = polyhedral::Scop::makeSpecializedScop(
      *scopTmp,
      globalParameterContext.intersect(scopTmp->globalParameterContext));
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << MappingOptions(execInfo_.options);
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << *(scopTmp->scheduleRoot());

  // Now we can build stuff
  auto mappedScop =
      polyhedral::MappedScop::makeWithOuterBlockInnerThreadStrategy(
          std::move(scopTmp), MappingOptions(execInfo_.options));
  mappedScop->setGPUType(polyhedral::MappedScop::GPUType::OPENCL);
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "Mapped schedule:" << std::endl
                                      << *(mappedScop->schedule());

  execInfo_.kernelParams = narrowParamsVector(
      mappedScop->scop().getParameterValues(globalParameterContext));
  kernelSpecializedName =
      specializeKernelName(execInfo_.kernelName, execInfo_.kernelParams);

  // This updates the launch bounds with the actual result from compilation
  // with tightening of launch_bounds.
  // What you get is not what you asked for, the autotuner should adapt to
  // that.
  // TODO mappedScop->setGPUType(GPUType::OPENCL);
  std::tie(openclSource, grid, block) =
      mappedScop->codegen(kernelSpecializedName);
  LOG_IF(INFO, FLAGS_dump_cuda) << "generatedOpencl: " << openclSource;
}

Duration OpenCLTcExecutor::run(
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs,
    bool profile) const {

  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);

  cl::Platform default_platform=all_platforms[0];

  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  cl::Device default_device=all_devices[1];

  //cl::Context context{default_device};
  int err;
  cl::Context context = rtcFun->program.getInfo<CL_PROGRAM_CONTEXT>(&err);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Couldn't recover context.";
  }
  cl::CommandQueue queue(context, default_device);

  auto src = rtcFun->program.getInfo<CL_PROGRAM_NUM_KERNELS>(&err);

  auto kernel = cl::Kernel(rtcFun->program, kernelSpecializedName.c_str(), &err);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Error with kernel: " << err << rtcFun->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device);
  }
  // Params
  for (auto i = 0; i < execInfo_.kernelParams.size(); ++i) {
    kernel.setArg(i, execInfo_.kernelParams[i]);
  }
  // Outputs
  std::vector<cl::Buffer> output_bufs;
  for (auto i = 0; i < outputs.size(); ++i) {
    auto& output = outputs[i];
    size_t size = 1;
    for (auto j = 0; j < output->ndim; ++j) {
      size *= output->shape[j];
    }
    output_bufs.emplace_back(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);
    kernel.setArg(execInfo_.kernelParams.size() + i, output_bufs[i]);
  }

  // convert these to opencl tensors
  std::vector<cl::Buffer> input_bufs;
  for (auto i = 0; i < inputs.size(); ++i) {
    auto& input = inputs[i];
    size_t size = 1;
    for (auto j = 0; j < input->ndim; ++j) {
      size *= input->shape[j];
    }
    input_bufs.emplace_back(context, CL_MEM_READ_ONLY, sizeof(float) * size);
    queue.enqueueWriteBuffer(input_bufs[i], CL_TRUE, 0, sizeof(float) * size, input->data);

    kernel.setArg(execInfo_.kernelParams.size() + outputs.size() + i, input_bufs[i]);
  }
  // run the program
  cl::Event event;
  queue.enqueueNDRangeKernel(kernel,
      cl::NDRange(0,0),
      cl::NDRange(grid[0] * block[0], grid[1] * block[1], grid[2] * block[2]),
      cl::NDRange(block[0], block[1], block[2]),
      NULL,
      &event);

  for (auto i = 0; i < outputs.size(); ++i) {
    auto& output = outputs[i];
    size_t size = 1;
    for (auto j = 0; j < output->ndim; ++j) {
      size *= output->shape[j];
    }
    queue.enqueueReadBuffer(output_bufs[i], CL_TRUE, 0, sizeof(float)*size, output->data);
  }
}

void OpenCLTcExecutor::uncheckedRun(
    const std::vector<const void*>& inputs,
    const std::vector<void*>& outputs) const {}

} // namespace tc
