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
    