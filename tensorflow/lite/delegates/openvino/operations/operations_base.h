#ifndef TENSORFLOW_LITE_DELEGATES_OPERATIONS_BASE_H_
#define TENSORFLOW_LITE_DELEGATES_OPERATIONS_BASE_H_

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset8.hpp>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/openvino/operations/openvino_node_manager.h"
#include "tensorflow/lite/tools/logging.h"


namespace tflite {
namespace openvinodelegate {

class OperationBuilder {
public:
    void UpdateNodeInfo(void* data, int size, void* builtin_data) {
        tensor_indices = (int*)data;
        tensor_indices_size = size;
        SetBuiltinData(builtin_data);
    }
    void SetContext(const TfLiteOpaqueContext* context) { context_ = context; }
    virtual std::shared_ptr<ov::Node> createNode() = 0;
    int* tensor_indices;
    int tensor_indices_size;
    std::shared_ptr<NodeManager> nodeManager;
    const TfLiteOpaqueContext* context_;

protected:
    // tflite runtime related info to be added in Model BUilder
    int operationIndex;
    void* GetBuiltinData() { return builtin_data_; }
    void* SetBuiltinData(void* builtin_data) { builtin_data_ = builtin_data; }
    std::shared_ptr<ov::Node> getInputNode(int index) {
        return nodeManager->getInterimNodeOutput(index);
    }
    std::shared_ptr<ov::Node> createConstNode(ov::element::Type elementType, ov::Shape shape,
                                              const void* data) {
        return std::make_shared<ov::opset8::Constant>(elementType, shape, data);
    }
    std::shared_ptr<ov::Node> ApplyActivation(std::shared_ptr<ov::Node> input,
                                              TfLiteFusedActivation activation) {
        // TODO: change activation type from Tflite to OV runtime
        switch (activation) {
            case kTfLiteActNone:
                return input;
            case kTfLiteActRelu:
                return std::make_shared<ov::opset8::Relu>(input);
            case kTfLiteActReluN1To1:
            case kTfLiteActRelu6:
                return std::make_shared<ov::opset8::Clamp>(input, 0, 6);
            case kTfLiteActTanh:
                return std::make_shared<ov::opset8::Tanh>(input);
            case kTfLiteActSignBit:
                return nullptr;
            case kTfLiteActSigmoid:
                return std::make_shared<ov::opset8::Sigmoid>(input);
            default:
                return nullptr;
        }
    }

    std::vector<int> GetDims(int index) {
        auto t = TfLiteOpaqueContextGetOpaqueTensor(context_, index);
        int32_t num_dims;
        num_dims = TfLiteOpaqueTensorNumDims(t);
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; i++) {
            dims[i]  = TfLiteOpaqueTensorDim(t,i);
        }
        return dims;
    }

private:
    void* builtin_data_ = nullptr;
    int op_type_ = 0;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPERATIOSN_BASE_H_
