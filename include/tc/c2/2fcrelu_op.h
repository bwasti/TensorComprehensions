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

#include <string>
#include <vector>

#include "tc/c2/tc_op.h"
#include "tc/library/2fcrelu.h"

namespace caffe2 {

template <typename T, class Context, class Engine = caffe2::DefaultEngine>
class Tc2FCReluOp : public TcOp<T, Context, Engine> {
 public:
  static constexpr auto description = tc::TC_2FCRELU;

  Tc2FCReluOp(const caffe2::OperatorDef& operator_def, caffe2::Workspace* ws)
      : TcOp<T, Context, Engine>(operator_def, ws) {
    this->tc_ = tc::TC_2FCRELU;
    this->tcName_ = tc::TC_2FCRELU_NAME;
  }

  ~Tc2FCReluOp() override {}

 protected:
  void setupNaiveMappingOptions() {
    this->mappingOptions_ = tc::MappingOptions::makeMlpMappingOptions();
  }
};
} // namespace caffe2
