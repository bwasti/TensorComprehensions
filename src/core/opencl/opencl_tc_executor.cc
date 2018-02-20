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

namespace tc {

void OpenCLTcExecutor::compile(const tc::MappingOptions& options) {
  if (rtcFun) {
    throw std::runtime_error{
        "CudaTcExecutor::compile cannot be called multiple tines."};
  }
  execInfo_.options = options.toProtobufSerializedString();
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
  std::tie(openclgSource, grid, block) =
      mappedScop->codegen(kernelSpecializedName);
  LOG_IF(INFO, FLAGS_dump_openclg) << "generatedOpencl: " << cudaSource;
}

Duration OpenCLTcExecutor::run(
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs,
    bool profile) const { }

void OpenCLTcExecutor::uncheckedRun(
    const std::vector<const void*>& inputs,
    const std::vector<void*>& outputs) const {}

} // namespace tc
