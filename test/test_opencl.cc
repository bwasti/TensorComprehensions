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
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>

#include <CL/cl.hpp>

#include "tc/core/constants.h"
#include "tc/core/libraries.h"
#include "tc/core/mapping_options.h"
#include "tc/core/polyhedral/codegen_gpu.h"
#include "tc/core/polyhedral/functional.h"
#include "tc/core/polyhedral/mapped_scop.h"
#include "tc/core/polyhedral/mapping_types.h"
#include "tc/core/polyhedral/schedule_isl_conversion.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_matcher.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/scope_guard.h"
#include "tc/external/isl.h"
#include "tc/library/matmul.h"
#include "tc/core/opencl/opencl_tc_executor.h"
#include "tc/core/opencl/opencl_execution_engine.h"

using namespace std;

using namespace tc;
using namespace tc::polyhedral;
using namespace tc::polyhedral::detail;

std::unique_ptr<Scop> Prepare(std::string tc) {
  auto ctx = isl::with_exceptions::globalIslCtx();
  // Build the SCoP corresponding to the Tc
  return Scop::makeScop(ctx, tc);
}

std::unique_ptr<Scop> PrepareAndJoinBands(std::string tc) {
  auto scop = Prepare(tc);
  // Join bands for ISL schedule tree to be tilable (set permutable flag
  // to true manually). This is generally incorrect and is only used for the
  // purpose of unit testing.
  joinBandsIterative(scop->scheduleRoot()->child({0}), true);
  return scop;
}
std::unique_ptr<MappedScop> makeUnmapped(std::string tc) {
  return MappedScop::makeOneBlockOneThread(Prepare(tc));
}

static MappingOptions DefaultOptions() {
  return MappingOptions::makeNaiveMappingOptions();
}

std::unique_ptr<MappedScop> TileAndMapThreads(
    std::unique_ptr<Scop>&& scop,
    const vector<size_t>& tileSizes,
    const array<size_t, 2>& blockSizes = {32ul, 8ul}) {
  // Keep non-const schedue tree pointer for testing purposes.
  auto root = scop->scheduleRoot();
  bandTile(root->child({0}), tileSizes, TileOptions::ShiftPointLoops);

  // Map to blocks (1 single block here)
  auto mscop = MappedScop::makeMappedScop(
      std::move(scop), Grid{1}, Block{blockSizes[0], blockSizes[1]}, 0, MappedScop::GPUType::OPENCL);
  USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
  auto band = mscop->map(root->child({0}), 0, BX);
  bandScale(band, tileSizes);

  auto ns = detail::ScheduleTree::collectDFSPostorder(
      root, detail::ScheduleTreeType::Band);
  mscop->map(ns[0], 0, TX);
  mscop->map(ns[1], 0, TY);
  mscop->insertMappingContext();
  return mscop;
}

std::unique_ptr<MappedScop> TileAndMapBlocksAndThreads(
    std::unique_ptr<Scop>&& scop,
    const vector<size_t>& tileSizes,
    const array<size_t, 2>& gridSizes,
    const array<size_t, 2>& blockSizes) {
  // Keep non-const schedue tree pointer for testing purposes.
  auto root = scop->scheduleRoot();
  bandTile(root->child({0}), tileSizes, TileOptions::ShiftPointLoops);
  auto mscop = MappedScop::makeMappedScop(
      std::move(scop),
      Grid{gridSizes[0], gridSizes[1]},
      Block{blockSizes[0], blockSizes[1]},
      0, MappedScop::GPUType::OPENCL);

  // Map to blocks
  USING_MAPPING_SHORT_NAMES(BX, BY, BZ, TX, TY, TZ);
  auto band = mscop->map(root->child({0}), 0, BX);
  band = mscop->map(band, 1, BY);
  bandScale(band, tileSizes);

  band = mscop->map(band->child({0}), 0, TX);
  band = mscop->map(band, 1, TY);
  mscop->insertMappingContext();
  return mscop;
}

cl::Program jitCompile(
    std::string opencl,
    cl::Context& context,
    std::vector<const char*> extraCompileOptions = std::vector<const char*>{}) {

    cl::Program::Sources sources;
    sources.push_back({opencl.c_str(), opencl.length()});

    // Compile
    cl::Program program(context, sources);
    std::vector<cl::Device> devs;
    auto err = context.getInfo(CL_CONTEXT_DEVICES, &devs);
    EXPECT_TRUE(err == CL_SUCCESS);
    err = program.build({devs[0]});
    EXPECT_TRUE(err == CL_SUCCESS) << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devs[0]);
    return program;
}

TEST(OpenCL, woot) {

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    EXPECT_TRUE(all_platforms.size() != 0);

    cl::Platform default_platform=all_platforms[0];
    LOG(INFO) << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    EXPECT_TRUE(all_devices.size() > 1); // GPU on device 1
    cl::Device default_device=all_devices[1];
    LOG(INFO) << "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

    cl::Context context{default_device};

/*
 void __global kernel hadamard(
     global const float* A,
     global const float* B,
     global float* C
   ) {
     //__global float (*A_)[16] = (__global float (*)[16])(A);
     int id = get_global_id(0);
     C[id] = A[id] * B[id];// + A_[0][1];
   }
*/
    auto program = jitCompile(R"OPENCL(
typedef int int32;
typedef long int64;
typedef float float32;
typedef double float64;


__global void __kernel hadamard_32_16(int32 M, int32 N, __global float32* poutput, __global float32* pA, __global float32* pB) {
  int b0 = get_group_id(0); int b1 = get_group_id(1); int b2 = get_group_id(2);
  int t0 = get_local_id(0); int t1 = get_local_id(1); int t2 = get_local_id(2);
  __global float32 (*output)[16] = (__global float32 (*)[16])(poutput);
  __global float32 (*A)[16] = (__global float32 (*)[16])(pA);
  __global float32 (*B)[16] = (__global float32 (*)[16])(pB);
  for (int c2 = t1; c2 <= 31; c2 += 8) {
    output[c2][t0] = (A[c2][t0]*B[c2][t0]);
  }
}
          )OPENCL", context);

    size_t N = 512;

    cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float) * N);
    cl::Buffer buffer_B(context, CL_MEM_READ_ONLY, sizeof(float) * N);
    cl::Buffer buffer_C(context, CL_MEM_WRITE_ONLY, sizeof(float) * N);

    float A[N], B[N], C[N];
    for (int i=0; i<N; i++) { A[i] = i; B[i] = N - i - 1; }

    cl::CommandQueue queue(context, default_device);
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float)*N, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(float)*N, B);

    auto hadamard = cl::Kernel(program, "hadamard");
    hadamard.setArg(0, 32);
    hadamard.setArg(1, 16);
    hadamard.setArg(2, buffer_C);
    hadamard.setArg(3, buffer_A);
    hadamard.setArg(4, buffer_B);
    cl::Event event;
    queue.enqueueNDRangeKernel(hadamard,
          cl::NullRange,
          cl::NDRange(1),
          cl::NDRange(1),
          NULL,
          &event);

    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float)*N, C);
    for (int i=0; i<N; i++) {
      ASSERT_NE(C[i], A[i] * B[i], 0.001);
    }
}

TEST(OpenCL, codegen) {
  string tc = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C) {
  C(i, j) = A(i, j) + B(i, j)
}
)TC";
  static constexpr auto specializedName = "kernel_anon";
  auto scop = PrepareAndJoinBands(tc);
  auto scopPtr = scop.get();
  auto context =
      scopPtr->makeContext(std::unordered_map<std::string, int>{{"N", 512}});
  scop = Scop::makeSpecializedScop(
      *scop, context.intersect(scop->globalParameterContext));
  auto mscop = TileAndMapBlocksAndThreads(
      std::move(scop), {16ul, 16ul}, {256ul, 256ul}, {16ul, 16ul});

  mscop->setGPUType(MappedScop::GPUType::OPENCL);

  auto res = mscop->codegen(specializedName);
  LOG(ERROR) << get<0>(res);
}

std::pair<std::vector<const DLTensor*>, std::vector<DLManagedTensor*>>
toConstDlpackTensors(const std::vector<at::Tensor>& tensors) {
  std::vector<const DLTensor*> dlTensors;
  std::vector<DLManagedTensor*> dlMTensors;
  for (auto tensor : tensors) {
    auto dlMTensor = at::toDLPack(tensor);
    dlTensors.push_back(&(dlMTensor->dl_tensor));
    dlMTensors.push_back(dlMTensor);
  }
  return make_pair(dlTensors, dlMTensors);
}

std::pair<std::vector<DLTensor*>, std::vector<DLManagedTensor*>>
toDlpackTensors(const std::vector<at::Tensor>& tensors) {
  std::vector<DLTensor*> dlTensors;
  std::vector<DLManagedTensor*> dlMTensors;
  for (auto tensor : tensors) {
    auto dlMTensor = at::toDLPack(tensor);
    dlTensors.push_back(&(dlMTensor->dl_tensor));
    dlMTensors.push_back(dlMTensor);
  }
  return make_pair(dlTensors, dlMTensors);
}

void deleteDlmTensors(std::vector<DLManagedTensor*>& tensors) {
  for (auto& tensor : tensors) {
    tensor->deleter(tensor);
  }
}

void prepareOutputs(
    lang::TreeRef func,
    const std::vector<const DLTensor*> tensorInfo,
    const at::Backend& backend,
    std::vector<at::Tensor>& outputs) {
  // prereqs for reusing CUDA memory, just allocate the first time then resize
  // (if needed). Most of the time should do nothing
  if (outputs.size() != 0 && outputs.size() != tensorInfo.size()) {
    throw lang::ErrorReport(func) << "expected " << tensorInfo.size()
                                  << " outputs but found " << outputs.size();
  }
  for (int i = 0; i < tensorInfo.size(); ++i) {
    auto info = tensorInfo[i];
    auto stype = at::toScalarType(info->dtype);
    if (outputs.size() < tensorInfo.size()) {
      outputs.push_back(at::getType(backend, stype)
                            .tensor(at::IntList(info->shape, info->ndim)));
      // TODO: we just malloc'ed I guess we can pay a memset
      outputs.back().zero_();
    } else {
      // In-place ATen operators have a trailing _
      std::vector<int64_t> shape(info->shape, info->shape + info->ndim);
      outputs[i].resize_(shape);
      // TODO: zero on shape increase? Not clear it's needed ..
      // outputs.back().zero_();
    }
  }
}


TEST(OpenCL, Basic) {
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);

  EXPECT_TRUE(all_platforms.size() != 0);

  cl::Platform default_platform=all_platforms[0];
  LOG(INFO) << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  EXPECT_TRUE(all_devices.size() > 1); // GPU on device 1
  cl::Device default_device=all_devices[1];
  LOG(INFO) << "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

  cl::Context context{default_device};

  OpenCLExecutionEngine exec;
  exec.define(
      R"(
    def hadamard(float(M,N) A, float(M,N) B) -> (float(M,N) output) {
      output(i,j) = A(i,j) * B(i,j)
    }
  )");
  at::Tensor a = at::CPU(at::kFloat).rand({32, 16});
  at::Tensor b = at::CPU(at::kFloat).rand({32, 16});
  at::Tensor c = at::CPU(at::kFloat).rand({32, 16});
  std::vector<at::Tensor> inputs = {a, b};
  std::vector<at::Tensor> outputs = {c};
  auto inputDLTensorsPair = toConstDlpackTensors(inputs);
  auto handle = exec.compile("hadamard", inputDLTensorsPair.first, tc::MappingOptions::makeNaiveMappingOptions());

  auto outTensorInfo =
      exec.inferOutputTensorInfo("hadamard", inputDLTensorsPair.first);
  auto input_vec = inputDLTensorsPair.first;
  prepareOutputs(
      exec.treeForFunction("hadamard"), outTensorInfo, at::Backend::CPU, outputs);
  auto outputDLTensorsPair = toDlpackTensors(outputs);
  auto output_vec = outputDLTensorsPair.first;

  ScopeGuard g1([&]() { deleteDlmTensors(inputDLTensorsPair.second); });
  ScopeGuard g2([&]() { deleteDlmTensors(outputDLTensorsPair.second); });
  exec.run(handle, input_vec, output_vec, false);
  auto gpu_data = (float*)output_vec[0]->data;
  for (auto i = 0; i < 10; ++i) {
    ASSERT_NE(gpu_data[i], a.data<float>()[i] * b.data<float>()[i], 0.001);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
