import os
import torch
import argparse
from halo import Halo
from utils import validatePath, loadClasses, ensureDir
from datasets import TrajectoryDataset
from torch.utils.data import DataLoader

def trainModel(classes_file, modeltype, dataset_file, input_size, pred_size, epochs, weights=None, output_path=None):
	device = torch.device('cpu')	
	if torch.cuda.is_available():
		device = torch.device('cuda')

	validatePath(classes_file)
	validatePath(dataset_file)

	classes = loadClasses(classes_file)
	for class_id, classname in enumerate(classes):
		dataset = TrajectoryDataset(dataset_file, class_id)
		dataloader = DataLoader(dataset, batch_size=64)
		
		if modeltype == "RNN":
			from predictors import RNNTrajectory
			model = RNNTrajectory(input_size, pred_size)
		elif modeltype == "LSTM":
			from predictors import LSTMTrajectory
			model = LSTMTrajectory(input_size, pred_size)
		elif modeltype == "GRU":			
			from predictors import GRUTrajectory
			model = GRUTrajectory(input_size, pred_size)
		
		model = model.to(device)

		if output_path is None:	
			dir_path = os.path.join(os.getcwd(), f"output/{classname}_{modeltype}")
		else:
			dir_path = os.path.join(output_path, f"{classname}_{modeltype}")
		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

		spinner = Halo(text="Loading data...", spinner="dots")
		spinner.start()
		model.train()
		average_loss = 0.0

		for epoch in range(epochs):
			running_loss = 0.0
			for i, data in enumerate(dataloader, 0):
				spinner.text = f"[{classname} : Epoch {(epoch + 1)}] Batch {i+1}"
				
				inputs, labels = data
				inputs = inputs.to(device)
				labels = labels.to(device)
    
				optimizer.zero_grad()
				
				outputs = model(inputs).to(device)
				loss = criterion(outputs, labels)
			
				loss.backward()
				optimizer.step()

				running_loss += loss.item()
				average_loss += running_loss
				if (i + 1) % 2000 == 0:    # print every 2000 mini-batches
					print(f"Epoch {epoch + 1} Batch {i + 1} Running Loss {running_loss / 2000}")
					running_loss = 0.0
			if (epoch + 1) % 1000 == 0:
				ensureDir(dir_path)
				torch.save(model.state_dict(), os.path.join(dir_path, f"{epoch + 1}.pth"))
    
		spinner.stop()

		ensureDir(dir_path)
		torch.save(model.state_dict(), os.path.join(dir_path, "final.pth"))

		model.eval()
		print(f"Average Loss for {classname} trained using {modeltype} is {round(average_loss / epochs, 5)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility Script to train models")
    parser.add_argument('--model', '-m', help="Model Name RNN / LSTM / GRU")
    parser.add_argument('--weights', '-w', default=None, help="Path to trained weights (if exist)")
    parser.add_argument('--dataset', '-d', help="Path to dataset file")
    parser.add_argument('--classes', '-c', help="Path to classes file")
    parser.add_argument('--input_size', '-i', default=8, help="Length of model input")
    parser.add_argument('--pred_size', '-p', default=12, help="Length of model output")
    parser.add_argument('--epochs', '-e', default=100, type=int, help="Number of epochs to train the model")
    parser.add_argument('--output', '-o', default=None, help="Path to store the trained model weights")
    args = parser.parse_args()
    
    trainModel(args.classes, args.model, args.dataset, args.input_size, args.pred_size, args.epochs, args.weights, args.output)