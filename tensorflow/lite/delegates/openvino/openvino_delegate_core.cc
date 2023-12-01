// openvino_graph_builder.cc
#include "openvino_delegate_core.h"

TfLiteStatus OpenVINODelegateManager::createGraphfromTfLite(
    TfLiteContext* context, const TfLiteDelegateParams* params) {
  openvino_graph_builder = std::make_unique<OpenVINOGraphBuilder>();

  std::unordered_set<int> outputs;
  for (int o = 0; o < params->output_tensors->size; o++) {
    const int output_tensor_idx = params->output_tensors->data[o];
    outputs.insert(output_tensor_idx);
  }
/*
  std::vector<int> tensors(context->tensors_size, -1);
  */
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
      const void* data = nullptr;
    if (context->tensors[t].allocation_type == kTfLiteMmapRo) {
      data = context->tensors[t].data.raw_const;
    }
    if (data == nullptr) {
      openvino_graph_builder->addInputParams(context, t);
    }
    else {

    }
      /*if (t >= 0) {      // node k has input of type int, value t
        tensors[t] = t;  // tensor[t] = t
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
    for (auto i = compute_inputs.begin(); i != compute_inputs.end(); i++) {
      if (context == nullptr) TFLITE_LOG(INFO) << "context is nullptr\n";
      openvino_graph_builder->addInputParams(context, *i);
    }
  }*/

    if (openvino_graph_builder->CreateNode(context, delegate_node_registration,
                                           delegate_node,
                                           delegate_node_id) != kTfLiteOk) {
      return kTfLiteError;
    }
  }

  openvino_graph_builder->resultNodes.resize(outputs.size());
  int it = 0;
  for (auto o : outputs) {
    openvino_graph_builder->resultNodes[it++] =
        openvino_graph_builder->getInterimNodefromTensor(o);
  }
  std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
      openvino_graph_builder->resultNodes, openvino_graph_builder->inputParams);
  ov::CompiledModel compiled_model;
  std::string deviceStr = "CPU";
  if (model) {
    compiled_model = openvino_delegate_core.compile_model(model, deviceStr);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>("/tmp/model.xml",
                                               "/tmp/model.bin");
    manager.run_passes(model);
  }

  inferRequest = compiled_model.create_infer_request();
  return kTfLiteOk;
}
