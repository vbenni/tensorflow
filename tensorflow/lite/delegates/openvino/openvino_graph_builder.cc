#include "openvino_graph_builder.h"

void OpenVINOGraphBuilder::setOutputAtOperandIndex(
    int index, ov::Output<ov::Node> output) {
  outputAtOperandIndex.insert({index, output});
}

std::shared_ptr<ov::Node> OpenVINOGraphBuilder::getInterimNodefromTensor(
    int index) {
  return std::shared_ptr<ov::Node>(
      outputAtOperandIndex.at(index).get_node_shared_ptr());
}

std::shared_ptr<ov::Node> OpenVINOGraphBuilder::createNodefromTensor(
    const TfLiteTensor& tensor, int tensor_index) {
  
  std::shared_ptr<ov::Node> input = getInterimNodefromTensor(tensor_index);
  return input;
}

std::shared_ptr<ov::Node> OpenVINOGraphBuilder::ApplyActivation(
    std::shared_ptr<ov::Node> input, TfLiteFusedActivation activation) {
  switch (activation) {
    case kTfLiteActNone:
      return input;
    case kTfLiteActRelu:
      return std::make_shared<ov::opset3::Relu>(input);
    case kTfLiteActReluN1To1:
    case kTfLiteActRelu6:
      return std::make_shared<ov::opset3::Clamp>(input, 0, 6);
    case kTfLiteActTanh:
      return std::make_shared<ov::opset3::Tanh>(input);
    case kTfLiteActSignBit:
      return nullptr;
    case kTfLiteActSigmoid:
      return std::make_shared<ov::opset3::Sigmoid>(input);
    default:
      return nullptr;
  }
}

TfLiteStatus OpenVINOGraphBuilder::CreateAddNode(
    TfLiteContext* context, int node_index, TfLiteNode* node,
    const TfLiteTensor* tensors,
    const TfLiteAddParams* add_params) {  // keep ptr const

  const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
  if (input1_tensor.type == kTfLiteFloat32) {
    ov::element::Type elementType = ov::element::f32;
    if (input1_tensor.allocation_type == kTfLiteMmapRo) {
      ov::Shape tensor_shape;
      const void* data = (const void*)input1_tensor.data.raw_const;
      for (int i = 0; i < input1_tensor.dims->size; i++)
        tensor_shape.push_back(
            static_cast<size_t>(input1_tensor.dims->data[i]));
      std::shared_ptr<ov::Node> input1 = createConstNode(elementType, tensor_shape, data);
    }
  }
  return kTfLiteOk;
}
/*
const TfLiteTensor &input2_tensor = tensors[node->inputs->data[1]];
auto inputNode1 =
    createNodefromTensor(input1_tensor, node->inputs->data[0]);
if (inputNode1 == nullptr)
  TFLITE_LOG(INFO) << "input node 1 is null\n";
auto inputNode2 =
    createNodefromTensor(input2_tensor, node->inputs->data[1]);
if (inputNode2 == nullptr)
  TFLITE_LOG(INFO) << "input Node 2 is null\n";
auto addNode = std::make_shared<ov::opset8::Add>(
    inputNode1, inputNode2, ov::op::AutoBroadcastType::NUMPY);
resultNode = ApplyActivation(addNode, add_params->activation);
setOutputAtOperandIndex(node->outputs->data[0], resultNode);
return kTfLiteOk;
}*/

TfLiteStatus OpenVINOGraphBuilder::CreateConv2DNode(
    TfLiteContext* context, int node_index, TfLiteNode* node,
    const TfLiteTensor* tensors, const TfLiteConvParams* conv2dParams) {
  const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
  const TfLiteTensor& filter_tensor = tensors[node->inputs->data[1]];
  const TfLiteTensor& bias_tensor = tensors[node->inputs->data[2]];

  std::vector<size_t> strides;
  std::vector<std::ptrdiff_t> padding_begin, padding_end;
  std::vector<size_t> dilations;
  ov::op::PadType auto_pad;
  int filter_size = 0;
  int padding_top, padding_bottom, padding_left, padding_right = 0;

  if (filter_tensor.dims->size == 4) {
    int filter_size = filter_tensor.dims->data[1];
  } else {
    return kTfLiteOk;
  }
  if (conv2dParams->padding == 0) {
    TFLITE_LOG(INFO) << "Padding 0\n";
  } else if (conv2dParams->padding == 1) {
    auto_pad = ov::op::PadType::EXPLICIT;

    padding_top = filter_size / 2;
    padding_bottom = filter_size / 2;
    padding_left = filter_size / 2;
    padding_right = filter_size / 2;
  } else if (conv2dParams->padding == 2) {
    auto_pad = ov::op::PadType::VALID;
    int padding_top, padding_bottom, padding_left, padding_right = 0;
  }

  strides = {(size_t)conv2dParams->stride_height,
             (size_t)conv2dParams->stride_width};
  padding_begin = {padding_top, padding_left};
  padding_end = {padding_bottom, padding_right};
  dilations = {(size_t)conv2dParams->dilation_height_factor,
               (size_t)conv2dParams->dilation_width_factor};
  auto input_node = createNodefromTensor(input_tensor, node->inputs->data[0]);
  auto filter_node = createNodefromTensor(filter_tensor, node->inputs->data[1]);
  auto bias_node = createNodefromTensor(bias_tensor, node->inputs->data[2]);

  auto convNode = std::make_shared<ov::opset3::Convolution>(
      input_node, filter_node, ov::Strides(strides),
      ov::CoordinateDiff(padding_begin), ov::CoordinateDiff(padding_end),
      ov::Strides(dilations), auto_pad);
  setOutputAtOperandIndex(node->outputs->data[0], convNode);
  return kTfLiteOk;
}

TfLiteStatus OpenVINOGraphBuilder::CreateNode(TfLiteContext* context,
                                              TfLiteRegistration* registration,
                                              TfLiteNode* node,
                                              int node_index) {
  switch (registration->builtin_code) {
    case kTfLiteBuiltinAdd: {
      const TfLiteAddParams* add_params =
          static_cast<const TfLiteAddParams*>(node->builtin_data);
      return CreateAddNode(context, node_index, node, context->tensors,
                           add_params);
    }
    case kTfLiteBuiltinConv2d: {
      const TfLiteConvParams* conv2d_params =
          static_cast<const TfLiteConvParams*>(node->builtin_data);
      return CreateConv2DNode(context, node_index, node, context->tensors,
                              conv2d_params);
    }
    default:
      return kTfLiteError;
  }
}