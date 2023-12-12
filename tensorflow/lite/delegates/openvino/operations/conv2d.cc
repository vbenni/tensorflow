#include "conv2d.h"

namespace tflite {
namespace openvinodelegate {

std::shared_ptr<ov::Node> Conv2D::createNode() {
	//TODO: filter tensor to update so it does not use tensors 
  const TfLiteTensor& filter_tensor = tensors[node->inputs->data[1]];


  const TfLiteConvParams* conv2dParams = (TfLiteConvParams*) GetBuiltinData(); 
  std::vector<size_t> strides;
  std::vector<std::ptrdiff_t> padding_begin, padding_end;
  std::vector<size_t> dilations;
  ov::op::PadType auto_pad;
  int filter_size = 0;
  int padding_top, padding_bottom, padding_left, padding_right = 0;

  if (filter_tensor.dims->size == 4) {
    filter_size = filter_tensor.dims->data[1];
  } else {
    return nullptr;
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
  auto input_node = getInputNode(0);
  auto filter_node = getInputNode(1);
  auto bias_node = getInputNode(2);

  auto convNode = std::make_shared<ov::opset8::Convolution>(
      input_node, filter_node, ov::Strides(strides),
      ov::CoordinateDiff(padding_begin), ov::CoordinateDiff(padding_end),
      ov::Strides(dilations), auto_pad);
  return convNode;
}

}  // namespace openvinodelegate
}  // namespace tflite
