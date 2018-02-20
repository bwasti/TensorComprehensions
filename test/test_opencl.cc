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

<<<<<<< e8b5736f87272dbd103fdd6737d263f7905bb772
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

    auto program = jitCompile(R"OPENCL(
          void kernel hadamard(
              global const float* A,
              global const float* B,
              global float* C
            ) {
              int id = get_global_id(0);
              C[id] = A[id] * B[id];
            }
          )OPENCL", context);

    size_t N = 1024;

    cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float) * N);
    cl::Buffer buffer_B(context, CL_MEM_READ_ONLY, sizeof(float) * N);
    cl::Buffer buffer_C(context, CL_MEM_WRITE_ONLY, sizeof(float) * N);

    float A[N], B[N], C[N];
    for (int i=0; i<N; i++) { A[i] = i; B[i] = N - i - 1; }

    cl::CommandQueue queue(context, default_device);
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float)*N, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(float)*N, B);

    auto hadamard = cl::Kernel(program, "hadamard");
    hadamard.setArg(0, buffer_A);
    hadamard.setArg(1, buffer_B);
    hadamard.setArg(2, buffer_C);
    cl::Event event;
    queue.enqueueNDRangeKernel(hadamard,
          cl::NullRange,
          cl::NDRange(100),
          cl::NullRange,
          NULL,
          &event);

    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float)*N, C);
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

  //mscop->setGPUType(MappedScop::GPUType::OPENCL);

  auto res = mscop->codegen(specializedName);
  LOG(ERROR) << get<0>(res);
}

TEST(OpenCL, Basic) {
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
