#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_CORE_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_CORE_H_
#include <vector>
#include <iostream>

#include "openvino/runtime/core.hpp"
#include "openvino_graph_builder.h"
class OpenVINODelegateManager {
 public:
  OpenVINODelegateManager(std::string_view plugins_path)
      : openvino_delegate_core(ov::Core(std::string(plugins_path))) {
    plugins_location = plugins_path;
  }
  TfLiteStatus openvino_delegate_init() {
    std::cout << plugins_location << std::endl;
    std::vector<std::string> ovDevices = openvino_delegate_core.get_available_devices();
    if (std::find(ovDevices.begin(), ovDevices.end(), "CPU") ==
        ovDevices.end()) {
      return kTfLiteDelegateError;
    } else {
      return kTfLiteOk;
    }
  }

  TfLiteStatus createGraphfromTfLite(TfLiteContext* context,
                                      const TfLiteDelegateParams* params);
  ov::InferRequest inferRequest;

 private:
  std::unique_ptr<OpenVINOGraphBuilder> openvino_graph_builder;
  ov::Core openvino_delegate_core;
  std::string plugins_location;
  std::shared_ptr<ov::Model> model;
  ov::CompiledModel compiled_model;
  std::string deviceStr = "CPU";
};
#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_KERNEL_H_