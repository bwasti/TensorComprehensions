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
#pragma once

#include "tc/core/mapping_options.h"
#include "tc/core/tc_executor.h"
#include <dlpack/dlpack.h>

namespace tc {

template <typename T>
class CLTensor {
  public:
   CLTensor(cl::Context& context, std::vector<int> dims, int access = CL_MEM_READ_WRITE) : context_(context) {
     size_ = 1;
     for (const auto& dim : dims) {
       size_ *= dim;
     }
     buf_ = cl::Buffer(context, access, sizeof(T) * size_);
   }
   bool write(const T* data) {
    std::vector<cl::Device> devs;
    auto err = context_.getInfo(CL_CONTEXT_DEVICES, &devs);
    if (err != CL_SUCCESS) { return false; }
    cl::CommandQueue queue(context_, devs[0]);
    queue.enqueueWriteBuffer(buf_, CL_TRUE, 0, sizeof(T) * size_, data);
  }
   bool read(T* data) {
    std::vector<cl::Device> devs;
    auto err = context_.getInfo(CL_CONTEXT_DEVICES, &devs);
    if (err != CL_SUCCESS) { return false; }
    cl::CommandQueue queue(context_, devs[0]);
    queue.enqueueReadBuffer(buf_, CL_TRUE, 0, sizeof(T) * size_, data);
  }
  private:
   size_t size_;
   cl::Buffer buf_;
   cl::Context& context_;
};

class OpenCLTcExecutor : public ::tc::TcExecutor {
 public:
  OpenCLTcExecutor(
      const std::string& def,
      const std::vector<const DLTensor*>& inputsInfo)
      : TcExecutor(def, inputsInfo) {}
  OpenCLTcExecutor(
      lang::TreeRef tree,
      const std::vector<const DLTensor*>& inputsInfo)
      : TcExecutor(tree, inputsInfo) {}
  ~OpenCLTcExecutor() {}

  OpenCLTcExecutor(OpenCLTcExecutor&&) = delete;
  OpenCLTcExecutor& operator=(OpenCLTcExecutor&&) = delete;
  OpenCLTcExecutor(const OpenCLTcExecutor&) = delete;
  OpenCLTcExecutor& operator=(const OpenCLTcExecutor&) = delete;

  void compile(const std::string& options) override {
    compile(MappingOptions(options));
  }
  void compile(const tc::MappingOptions& options);

  Duration run(
      const std::vector<const DLTensor*>& inputs,
      const std::vector<DLTensor*>& outputs,
      bool profile = false) const;

  void uncheckedRun(
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs) const;

  bool hasRTCFun() {
    return rtcFun.get() != nullptr;
  }

  std::string kernelName() const {
    return execInfo_.kernelName;
  }

 private:
  void compileWithTcMapper();

 public:
  std::string kernelSpecializedName;
  std::string openclSource;
  Grid grid{{0, 0, 0}};
  Block block{{0, 0, 0}};

 protected:
  std::shared_ptr<OpenCLRTCFunction> rtcFun;

};

} // namespace tc
