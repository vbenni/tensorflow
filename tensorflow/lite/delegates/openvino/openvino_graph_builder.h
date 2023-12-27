#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/openvino/operations/include/add.h"
#include "tensorflow/lite/delegates/openvino/operations/include/conv2d.h"
#include "tensorflow/lite/delegates/openvino/operations/openvino_node_manager.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace openvinodelegate {

class OpenVINOGraphBuilder {
public:
    OpenVINOGraphBuilder() { nodeManager = std::make_shared<NodeManager>(); }

    void addInputParams(const TfLiteOpaqueContext* context, const int index) {
        auto t = TfLiteOpaqueContextGetOpaqueTensor(context, index);
        int32_t num_dims;
        num_dims = TfLiteOpaqueTensorNumDims(t);
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; i++) {
            dims[i]  = TfLiteOpaqueTensorDim(t,i);
        }
        auto input = std::make_shared<ov::opset3::Parameter>(ov::element::f32,
                                                             ov::Shape(dims.begin(), dims.end()));
        if (input == NULL) TFLITE_LOG(INFO) << "addInputParams input node is null\n";
        nodeManager->setOutputAtOperandIndex(index, input);
        inputParams.push_back(input);
    }

    void createConstNode(const TfLiteOpaqueContext* context, const int index) {
        const TfLiteOpaqueTensor* t = TfLiteOpaqueContextGetOpaqueTensor(context, index);
        int32_t num_dims;
        num_dims = TfLiteOpaqueTensorNumDims(t);
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; i++) {
            dims[i]  = TfLiteOpaqueTensorDim(t,i);
        }
        const void* data = TfLiteOpaqueTensorData(t);
        auto constNode = std::make_shared<ov::opset8::Constant>(
            ov::element::f32, ov::Shape(dims.begin(), dims.end()), data);
        nodeManager->setOutputAtOperandIndex(index, constNode);
    }

    void updateResultNodes(std::vector<int> outputs) {
        for (auto o : outputs) {
            resultNodes.push_back(nodeManager->getInterimNodeOutput(o));
        }
    }

    std::vector<std::shared_ptr<ov::Node>> getResultNodes() { return resultNodes; }

    std::vector<std::shared_ptr<ov::opset3::Parameter>> getInputParams() { return inputParams; }

    TfLiteStatus createNodeFromTfLiteOp(int node_id, TfLiteRegistrationExternal* registration,
                                        TfLiteOpaqueNode* node, TfLiteOpaqueContext* context);
    std::shared_ptr<OperationBuilder> createOpClass(int operationIndex,
                                                    TfLiteRegistrationExternal* registration);
    std::vector<std::shared_ptr<ov::opset3::Parameter>> inputParams;
    std::vector<std::shared_ptr<ov::Node>> resultNodes;

private:
    std::shared_ptr<ov::Node> resultNode;
    std::shared_ptr<NodeManager> nodeManager;
};
}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
