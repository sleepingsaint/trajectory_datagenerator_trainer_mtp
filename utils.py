import os
import sys
import onnxruntime
import torch

from datasets import TrajectoryDataset


def validatePath(path):
    # checking if the given file path exists or not
    try:
        assert os.path.exists(path) and os.path.isfile(path)
    except AssertionError:
        sys.exit(f"Invalid Path {path}")


def loadClasses(path):
    validatePath(path)
    classes = []
    with open(path, "r") as f:
        classes = f.read().split("\n")

    return classes


def ensureDir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def splitDataset(dataset, ratio=0.2):
    dataset_len = len(dataset)
    valid_len = int(dataset_len * ratio)
    if valid_len == 0:
        return dataset, dataset
    train_len = dataset_len - valid_len
    return torch.utils.data.random_split(dataset, [train_len, valid_len])


def convertOnnxModel(model, weights_path, output, input_tensor=torch.rand((1, 8, 2))):
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    torch.onnx.export(
        model,
        input_tensor,
        output,
        export_params=True,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        })

    return checkOnnxModel(output)


def checkOnnxModel(model_path):
    import onnx
    onnx_model = onnx.load(model_path)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
        return False

    print('The model is valid!')
    return True


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def load_models(dir_path, classes_path, model_type):
    classes = loadClasses(classes_path)
    ort_sessions = []
    for classname in classes:
        model_path = os.path.join(dir_path, f"{classname}_{model_type}.onnx")
        ort_session = onnxruntime.InferenceSession(model_path, providers=["CUDAExecutionProvider", ])
        ort_sessions.append(ort_session)

    return ort_sessions


def getMeanAndVar(dataset_file, class_idx):
    dataset = TrajectoryDataset(dataset_file, float(class_idx))
    return (dataset.getMean(), dataset.getVar())
