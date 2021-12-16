import os
import sys

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

def splitDataset(dataset, ratio = 0.2):
    import torch
    dataset_len = len(dataset)
    valid_len = int(dataset_len * ratio)
    if valid_len == 0:
        return dataset, dataset
    train_len = dataset_len - valid_len
    return  torch.utils.data.random_split(dataset, [train_len, valid_len]) 
