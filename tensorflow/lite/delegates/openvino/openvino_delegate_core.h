#include <openvino/runtime/core.hpp>
#include <vector>

#include "openvino_builder.h"
class OpenVINODelegateManager {
 public:
  OpenVINODelegateManager() {}

  TfLiteStatus openvino_delegate_init() {
    ov::Core core(plugins_location);
    std::vector<std::string> ovDevices = core.get_available_devices();
    if (std::find(ovDevices.begin(), ovDevices.end(), "CPU") ==
        ovDevices.end()) {
      return kTfLiteDelegateError;
    } else {
      return kTfLiteOk;
    }
  }

  TfLiteStatus setTfLiteGraphInfo(TfLiteContext* context,
                                  const TfLiteDelegateParams* params) {
    openvino_graph_builder->setGraphInputs(
        &params->input_tensors->data[0],
        &params->input_tensors->data[params->input_tensors->size]);

    for (int o = 0; o < params->output_tensors->size; o++) {
      const int output_tensor_idx = params->output_tensors->data[o];
      outputs.insert(output_tensor_idx);
    }
  }

  std::vector<int> tensors(context->tensors_size, -1);
  for (int i = 0; i < params->nodes_to_replace->size; i++) {
    const int delegate_node_id = params->nodes_to_replace->data[i];
    TfLiteNode* delegate_node;
    TfLiteRegistration* delegate_node_registration;
    if (context->GetNodeAndRegistration(context, delegate_node_id,
                                        &delegate_node,
                                        &delegate_node_registration))
      return kTfLiteError;

    for (int k = 0; k < delegate_node->inputs->size; k++) {
      if (delegate_node_registration->builtin_code ==
              kTfLiteBuiltinTransposeConv &&
          k == 0) {
        continue;
      }
      const int t = delegate_node->inputs->data[k];
      if (t >= 0) {
        tensors[t] = t;
      }
    }
    for (int k = 0; k < delegate_node->outputs->size; k++) {
      const int t = delegate_node->outputs->data[k];
      if (t >= 0) {
        tensors[t] = t;
      }
    }
  }

  // Filter out and remove -1 (unused) indexes.
  tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                               [](int i) { return i < 0; }),
                tensors.end());
  std::sort(tensors.begin(), tensors.end());

  for (int t : tensors) {
    const void* data = nullptr;
    if (context->tensors[t].allocation_type == kTfLiteMmapRo) {
      data = context->tensors[t].data.raw_const;
    }
    if (inputs.count(t) != 0) {
      if (data == nullptr) {
        compute_inputs.insert(t);
      }
    }
  }
}

private : std::shared_ptr<OpenVINOGraphBuilder>
          openvino_graph_builder();

std::string plugins_location =
    "/home/adattatr/openvino_install/openvino/runtime/lib/intel64/plugins.xml"
}