import argparse
import numpy as np
from pathlib import Path

import onnx
import torch.onnx
import onnxruntime
import onnx_coreml

from argus import load_model

from src import config

config.kernel = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch model to CoreML')

    parser.add_argument(
        '-m', '--model_path', required=True,
        help='Path to torch model', type=str)
    parser.add_argument(
        '-d', '--save_dir', required=True,
        help='Directory for output saving', type=str)
    parser.add_argument(
        '-n', '--save_name', required=True,
        help='Output model name', type=str)

    args = parser.parse_args()
    return args


def get_sample_input():
    batch_size = 1
    return torch.randn(batch_size, 1, 128, 256, requires_grad=True)


def load_torch_model(model_path):
    model = load_model(model_path, device='cpu')
    print("Model params:", model.params)
    torch_model = model.nn_module
    torch_model.return_aux = False
    torch_model.eval()
    return torch_model


def convert_pytorch_to_onnx(torch_model, save_path):
    x = get_sample_input()
    torch_out = torch_model(x)
    print("Output shape", torch_out.shape)

    # Export the model
    torch.onnx.export(torch_model,
                      x,
                      save_path,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['spectrogram'],
                      output_names=['classLabelLogits'])


def check_onnx_model(torch_model, model_path):
    # verify the model’s structure and confirm that the model has a valid schema
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    # check that output of PyTorch and ONNX Runtime runs match numerically
    x = get_sample_input()
    torch_out = torch_model(x)

    ort_session = onnxruntime.InferenceSession(str(model_path))

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute onnxruntime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare onnxruntime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def convert_onnx_to_coreml(onnx_model_path, coreml_model_path, class_labels_path):
    # onnx model --> Apple Core ML
    mlmodel = onnx_coreml.convert(onnx.load(onnx_model_path),
                                  # image_input_names=['spectrogram'],
                                  mode='classifier',
                                  class_labels=str(class_labels_path))

    mlmodel.author = 'Ruslan Baikulov'
    mlmodel.license = 'MIT'
    mlmodel.short_description = 'This model takes a mel-spectrogram and predicts sound events'
    mlmodel.input_description['spectrogram'] = 'Mel-spectrogram 128x256'
    mlmodel.output_description['classLabelLogits'] = "Sound event logits"
    mlmodel.output_description['classLabel'] = 'Sound event name'

    mlmodel.save(coreml_model_path)


if __name__ == "__main__":
    args = parse_arguments()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    onnx_model_path = save_dir / f'{args.save_name}.onnx'
    coreml_model_path = save_dir / f'{args.save_name}.mlmodel'
    class_labels_path = save_dir / f'{args.save_name}_labels.txt'

    with open(class_labels_path, 'w') as file:
        for cls in config.classes:
            file.write(f"{cls}\n")

    torch_model = load_torch_model(args.model_path)
    convert_pytorch_to_onnx(torch_model, onnx_model_path)
    check_onnx_model(torch_model, onnx_model_path)
    convert_onnx_to_coreml(onnx_model_path,
                           coreml_model_path,
                           class_labels_path)
