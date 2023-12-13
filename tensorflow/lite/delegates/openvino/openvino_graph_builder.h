#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/delegates/openvino/operations/openvino_node_manager.h"
#include "tensorflow/lite/delegates/openvino/operations/add.h"
#include "tensorflow/lite/delegates/openvino/operations/conv2d.h"

namespace tflite {
namespace openvinodelegate {

class OpenVINOGraphBuilder {
 public:
  OpenVINOGraphBuilder(int size) {
  // TODO: update later if list of nodes is required    operationNodes.resize(size);
      nodeManager = std::make_shared<NodeManager>(size + 1 );
  }

  void addInputParams(const TfLiteContext* context, const int index) {
    const TfLiteTensor t = context->tensors[index];
    std::vector<size_t> dims(t.dims->size);
    for (int i = 0; i < t.dims->size; i++) {
      dims[i] = t.dims->data[i];
    }
    auto input = std::make_shared<ov::opset3::Parameter>(
        ov::element::f32, ov::Shape(dims.begin(), dims.end()));
    if (input == NULL)
      TFLITE_LOG(INFO) << "addInputParams input node is null\n";
    nodeManager->setOutputAtOperandIndex(index, input);
    inputParams.push_back(input);
  }
  std::shared_ptr<ov::Node> createConstNode(const TfLiteContext* context, const int index) {
    const TfLiteTensor t = context->tensors[index];
    std::vector<size_t> dims(t.dims->size);
    for (int i = 0; i < t.dims->size; i++) {
      dims[i] = t.dims->data[i];
    }
    const void *data = (const void *) t.data.raw_const;
    auto constNode = std::make_shared<ov::opset8::Constant>(ov::element::f32, ov::Shape(dims.begin(), dims.end()), data);
    nodeManager->setOutputAtOperandIndex(index, constNode);
  }

  void updateResultNodes(std::unordered_set<int> outputs) {
      for(auto o : outputs) {
         resultNodes.push_back(nodeManager->getInterimNodeOutput(o));
      }
  }

  std::vector<std::shared_ptr<ov::Node>> getResultNodes() {
      return resultNodes;
  }

  std::vector<std::shared_ptr<ov::opset3::Parameter>> getInputParams() {
      return inputParams;
  }

  TfLiteStatus createNodeFromTfLiteOp(int node_id, TfLiteRegistration* registration, TfLiteNode* node, TfLiteContext* context);
  std::shared_ptr<OperationBuilder> createOpClass(int operationIndex, TfLiteRegistration* registration);
  std::vector<std::shared_ptr<ov::opset3::Parameter>> inputParams;
  std::vector<std::shared_ptr<ov::Node>> resultNodes;

 private:
  std::shared_ptr<ov::Node> resultNode;
  std::shared_ptr<NodeManager> nodeManager;
};
}  // namespace openvinodelegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
