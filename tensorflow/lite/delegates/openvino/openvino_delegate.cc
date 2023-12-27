/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "openvino_delegate.h"

#include <ie_cnn_network.h>

#include "openvino/runtime/core.hpp"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace tflite {
namespace openvinodelegate {
class OpenVINODelegate : public SimpleOpaqueDelegateInterface {
public:
    explicit OpenVINODelegate(const TfLiteOpenVINODelegateOptions* options) {
        options_ = *options;
        if (options == nullptr) options_ = TfLiteOpenVINODelegateOptionsDefault();
    }

    bool CheckInputsType(int tensor_id, TfLiteOpaqueContext* context, TfLiteType expected_type) const {
        const TfLiteOpaqueTensor* opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
        TfLiteType type = TfLiteOpaqueTensorType(opaque_tensor);
        return expected_type == type;
    }
    bool CheckNodeSupportByOpenVINO(const TfLiteRegistrationExternal* registration, const TfLiteOpaqueNode* node,
                                    TfLiteOpaqueContext* context) const {
        switch (TfLiteRegistrationExternalGetBuiltInCode(registration)) {
            case kTfLiteBuiltinAdd: {
                const int* inputs;
                int num_inputs;
                auto tfStatus = TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs);
                int tensor_id1 = inputs[0];
                int tensor_id2 = inputs[1];
                if (CheckInputsType(tensor_id1, context, kTfLiteFloat32) &&
                    CheckInputsType(tensor_id2, context, kTfLiteFloat32))
                    return true;
                return false;
            }
            case kTfLiteBuiltinConv2d: {
                return true;
            }
            default:
                return false;
        }
    }

    bool IsNodeSupportedByDelegate(const TfLiteRegistrationExternal* registration, const TfLiteOpaqueNode* node,
                                   TfLiteOpaqueContext* context) const override {
        bool check = CheckNodeSupportByOpenVINO(registration, node, context);
        return check;
    }

    TfLiteStatus Initialize(TfLiteOpaqueContext* context) override { return kTfLiteOk; }

    const char* Name() const override { return "OpenVINO SimpleOpaqueDelegate"; }

    std::unique_ptr<SimpleOpaqueDelegateKernelInterface> CreateDelegateKernelInterface() override {
        return std::unique_ptr<OpenVINODelegateKernel>(
            new tflite::openvinodelegate::OpenVINODelegateKernel());
    }

private:
    TfLiteOpenVINODelegateOptions options_;
};
}  // namespace openvinodelegate
}  // namespace tflite

TfLiteDelegate* TFL_CAPI_EXPORT
TfLiteCreateOpenVINODelegate(const TfLiteOpenVINODelegateOptions* options) {
    auto ovdelegate_ = std::make_unique<tflite::openvinodelegate::OpenVINODelegate>(options);
    return tflite::TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(std::move(ovdelegate_));
}

void TFL_CAPI_EXPORT TfLiteDeleteOpenVINODelegate(TfLiteDelegate* delegate) { return; }

TfLiteOpenVINODelegateOptions TFL_CAPI_EXPORT TfLiteOpenVINODelegateOptionsDefault() {
    TfLiteOpenVINODelegateOptions result;
    result.debug_level = 0;
    result.plugins_path = "";
    return result;
}
