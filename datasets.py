import torch
import pandas as pd
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, data_file, class_id, input_len = 8, pred_len = 12) -> None:
        """
        input_len: length of input given to the model
        pred_len: length of predictions to predict
        """     
        
        super(TrajectoryDataset, self).__init__()
        data = pd.read_csv(data_file)
        # getting all the class coordinates
        # finding the diff with previous positions
        # excluding the first row since it will be (NaN, NaN)
        class_data = data.loc[data['class_id'] == float(class_id)][['x', 'y']].diff()
        self.dataframe = class_data[1:] 
        self.input_len = input_len
        self.pred_len = pred_len
        self.block_len = input_len + pred_len 

    def __len__(self):
        return len(self.dataframe) // self.block_len 
    
    def __getitem__(self, idx):
        items = self.dataframe[['x', 'y']].iloc[idx:idx+self.block_len]
        np_item = [x for x in items.to_numpy()]
        inputs = (torch.Tensor(np_item[0:self.input_len]) - self.getMean()) / self.getStd()
        outputs = (torch.Tensor(np_item[self.input_len:]) - self.getMean()) / self.getStd()

        return inputs, outputs
    
    def getMean(self):
        return torch.Tensor([self.dataframe['x'].mean(), self.dataframe['y'].mean()])
    
    def getVar(self):
        return torch.Tensor([self.dataframe['x'].var(), self.dataframe['y'].var()])
    
    def getStd(self):
        return torch.Tensor([self.dataframe['x'].std(), self.dataframe['y'].std()])

if __name__ == "__main__":
    dataset = TrajectoryDataset("1920_1080_probe.csv", 1)
    print(dataset[0])
    