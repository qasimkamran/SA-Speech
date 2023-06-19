import onnx
import onnxruntime
import onnx2keras
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def torch_to_onnx(torch_model, input_shape, filename):
    model.eval() 
    dummy_input = torch.randn(1, input_shape, requires_grad=True)  
    torch.onnx.export(torch_model,               # model being run
                      dummy_input,               # model input (or a tuple for multiple inputs)
                      filename,                  # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      input_names = ['input'],   # the model's input names
                      output_names = ['output']) # the model's output names
    onnx_load(filename)
    onnx.checker.check_model(onnx_model)
    onnx_model_summary(onnx_model)
    return onnx_model


def onnx_model_summary(onnx_model):
    for i, layer in enumerate(onnx_model.graph.node):
        print(f"Layer {i + 1}: {layer.name}")
        print(f"Inputs: {layer.input}")
        print(f"Output: {layer.output}")


def onnx_to_keras(onnx_model, name_policy='renumerate'):
    keras_model = onnx_to_keras(onnx_model, ['input'], name_policy='renumerate')
    keras_model.summary()
    return keras_model
