#include "openvino_graph_builder.h"

TfLiteStatus OpenVINOGraphBuilder::setNodeAtTensorIndex(
    int index, ov::Output<ov::Node> output) {
  if (index < 0 || output.get_node_shared_ptr() == nullptr) {
    return kTfLiteDelegateError;
  }
  TFLITE_LOG(INFO) << "index for set = " << index 
                   << "\n";
  outputAtOperandIndex.insert({index, output});
  return kTfLiteOk;
}

std::shared_ptr<ov::Node> OpenVINOGraphBuilder::getOVNodefromTensorIndex(
    int index) {
  TFLITE_LOG(INFO) << "index for get = " << index 
                   << "\n";
  return std::shared_ptr<ov::Node>(
      outputAtOperandIndex.at(index).get_node_shared_ptr());
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

  // Create OV Nodes for Constant Nodes
  // Interim

  auto addinputNode1 = getOVNodefromTensorIndex(node->inputs->data[0]);
  auto addinputNode2 = getOVNodefromTensorIndex(node->inputs->data[1]);

  auto addNode = std::make_shared<ov::opset8::Add>(
      addinputNode1, addinputNode2, ov::op::AutoBroadcastType::NUMPY);
  resultNode = ApplyActivation(addNode, add_params->activation);
  setNodeAtTensorIndex(node->outputs->data[0], resultNode);
  return kTfLiteOk;
}

std::shared_ptr<ov::Node> OpenVINOGraphBuilder::transpose(ConversionType type,
                                                    ov::Output<ov::Node> input) {
    ov::AxisVector order;
    switch (type) {
        case NHWC_NCHW:
            order = {0, 3, 1, 2};
            break;
        case NCHW_NHWC:
            order = {0, 2, 3, 1};
            break;
        case IHWO_OIHW:
            order = {3, 0, 1, 2};
            break;
        case OHWI_OIHW:
            order = {0, 3, 1, 2};
            break;
        case NHWC_CWHN:
            order = {3, 2, 1, 0};
            break;
        case CWHN_NHWC:
            order = {3, 2, 1, 0};
            break;
        case NHC_NCH:
            order = {0, 2, 1};
            break;
        case NCH_NHC:
            order = {0, 1, 2};
            break;
        case CNH_NHC:
            order = {1, 2, 0};
            break;
        case NHC_CNH:
            order = {2, 0, 1};
            break;
        case BTS_TBS:
            order = {1, 0, 2};
            break;
        case NHCW_NHWC:
            order = {0, 1, 3, 2};
            break;
        case NC_CN:
            order = {1, 0};
            break;
        default:
            //ALOGE("Invalid transpose operation !!");
            break;
    }
    const auto order_node =
        ov::opset3::Constant::create(ov::element::i64, ov::Shape{order.size()}, order);
    return std::make_shared<ov::opset3::Transpose>(input, order_node);
}


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
    filter_size = filter_tensor.dims->data[1];
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
  auto input_node = getOVNodefromTensorIndex(node->inputs->data[0]);
  input_node = transpose(NHWC_NCHW, input_node);
  auto filter_node = transpose(OHWI_OIHW, getOVNodefromTensorIndex(node->inputs->data[1]));
  auto bias_node = getOVNodefromTensorIndex(node->inputs->data[2]);

  auto convNode = std::make_shared<ov::opset3::Convolution>(
      input_node, filter_node, ov::Strides(strides),
      ov::CoordinateDiff(padding_begin), ov::CoordinateDiff(padding_end),
      ov::Strides(dilations), auto_pad);
  std::shared_ptr<ov::Node> conv_node_output = transpose(NCHW_NHWC, convNode);
  setNodeAtTensorIndex(node->outputs->data[0], conv_node_output);
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