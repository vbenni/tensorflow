#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/logging.h"

class OpenVINOGraphBuilder {
 public:
  OpenVINOGraphBuilder() {}

 public:
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
    setOutputAtOperandIndex(index, input);
    inputParams.push_back(input);
  }
  // REVISIT: Decide on the data type of index passed in these calls
  void setOutputAtOperandIndex(int index, ov::Output<ov::Node> output);
  std::shared_ptr<ov::Node> getInterimNodefromTensor(int index);
  std::shared_ptr<ov::Node> createNodefromTensor(const TfLiteTensor& tensor,
                                         int tensor_index);
  std::shared_ptr<ov::Node> createConstNode(ov::element::Type elementType,
                                            ov::Shape shape, const void* data) {
    return std::make_shared<ov::opset8::Constant>(elementType, shape, data);
  }

  // Ops

  std::shared_ptr<ov::Node> ApplyActivation(std::shared_ptr<ov::Node> input,
                                            TfLiteFusedActivation activation);
  TfLiteStatus CreateAddNode(TfLiteContext* context, int node_index,
                             TfLiteNode* node, const TfLiteTensor* tensors,
                             const TfLiteAddParams* add_params);

  TfLiteStatus CreateConv2DNode(TfLiteContext* context, int node_index,
                                TfLiteNode* node, const TfLiteTensor* tensors,
                                const TfLiteConvParams* conv2dParams);

  TfLiteStatus CreateNode(TfLiteContext* context,
                          TfLiteRegistration* registration, TfLiteNode* node,
                          int node_index);
  std::vector<std::shared_ptr<ov::Node>> resultNodes = {};
  std::vector<std::shared_ptr<ov::opset3::Parameter>> inputParams;

 private:
  std::map<int, ov::Output<ov::Node>> outputAtOperandIndex;
  std::shared_ptr<ov::Node> resultNode;
};
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_GRAPH_BUILDER_H_