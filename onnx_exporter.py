import os
import torch
import argparse
from utils import loadClasses, convertOnnxModel, ensureDir

def validateDir(path):
	return os.path.exists(path) and os.path.isdir(path)

def export(weights_dir, classes_file, model_types, output_dir):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    assert validateDir(weights_dir)
    ensureDir(output_dir)
    classes = loadClasses(classes_file)
    for model_type in model_types:
        for classname in classes:
            if model_type == "RNN":
                from predictors import RNNTrajectory
                model = RNNTrajectory(8, 12, device)
            elif model_type == "LSTM":
                from predictors import LSTMTrajectory
                model = LSTMTrajectory(8, 12, device)
            else:
                from predictors import GRUTrajectory
                model = GRUTrajectory(8, 12, device)
            
            weights_path = os.path.join(weights_dir, f"{classname}_{model_type}", "final.pth")
            output_path = os.path.join(output_dir, f"{classname}_{model_type}.onnx")
            convertOnnxModel(model, weights_path, output_path)

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helper script to convert pytorch model to onnx format")
    parser.add_argument('-c', '--classes', help="Path to the classes file")
    parser.add_argument('-w', '--weights', help="Path to the pytorch weights directory")
    parser.add_argument('-m', '--model', nargs='+', help="Type of the model RNN / LSTM / GRU")
    parser.add_argument('-o', '--output', help="Path to the output directory")
    
    args = parser.parse_args()
    export(args.weights, args.classes, args.model, args.output)
