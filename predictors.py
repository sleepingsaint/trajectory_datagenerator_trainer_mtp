import torch
import torch.nn as nn

class RNNTrajectory(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers = 2, batch_first=True):
        """
        input_size - number of time intervals over which data is taken
         
        hidden_layer_size - denotes the output size
        
        num_layers - number of RNN layers to stack upon each other
        
        sequence length - can be anything, but keeping it constant for the whole dataset is suggested
        """
        super(RNNTrajectory, self).__init__()
         
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_layer_size
        self.batch_first = batch_first
        
        self.rnn = nn.RNN(input_size, hidden_layer_size, num_layers, batch_first=batch_first)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor :
        """
        input shape - [Batch Size, Input Size, Sequence Length]
        
        RNN output shape - [Batch Size, Sequence Length, Hidden Size]
        
        Return Output Shape - [Batch Size, Hidden Size, Sequence Length]
        
        h0 - initial hidden - initialized to zeroes
        """
        input = torch.transpose(input, 1, 2)
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        out, _ = self.rnn(input, h0)
        output = torch.transpose(out, 1, 2)
        
        return output 
    
class LSTMTrajectory(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers = 2, batch_first=True):
        """
        input_size - number of time intervals over which data is taken
         
        hidden_layer_size - denotes the output size
        
        num_layers - number of RNN layers to stack upon each other
        
        sequence length - can be anything, but keeping it constant for the whole dataset is suggested
        """
        super(LSTMTrajectory, self).__init__()
         
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_layer_size
        self.batch_first = batch_first
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=batch_first)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input shape - [Batch Size, Input Size, Sequence Length]
        
        RNN output shape - [Batch Size, Sequence Length, Hidden Size]
        
        Return Output Shape - [Batch Size, Hidden Size, Sequence Length]
        
        h0 - initial hidden state - initialized to zeros
        
        c0 - initial cell state - initialized to zeros
        """
        input = torch.transpose(input, 1, 2)
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        
        out, _ = self.lstm(input, (h0, c0))
        output = torch.transpose(out, 1, 2)
        
        return output 


class GRUTrajectory(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers = 2, batch_first=True):
        """
        input_size - number of time intervals over which data is taken
         
        hidden_layer_size - denotes the output size
        
        num_layers - number of RNN layers to stack upon each other
        
        sequence length - can be anything, but keeping it constant for the whole dataset is suggested
        """
        super(GRUTrajectory, self).__init__()
         
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_layer_size
        self.batch_first = batch_first
        
        self.gru = nn.GRU(input_size, hidden_layer_size, num_layers, batch_first=batch_first)
    
    def forward(self, input) -> torch.Tensor:
        """
        input shape - [Batch Size, Input Size, Sequence Length]
        
        RNN output shape - [Batch Size, Sequence Length, Hidden Size]
        
        Return Output Shape - [Batch Size, Hidden Size, Sequence Length]
        
        h0 - initial hidden state - initialized to zeros
        """
        input = torch.transpose(input, 1, 2)
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        out, _ = self.gru(input, h0)
        output = torch.transpose(out, 1, 2)
        
        return output 


if __name__ == "__main__":
    rnnTrajectory = RNNTrajectory(12, 7)
    lstmTrajectory = LSTMTrajectory(12, 7)
    gruTrajectory = GRUTrajectory(12, 7)

    input = torch.tensor([[[x, x] for x in range(12)]], dtype=torch.float)
    print(input.size()) # torch.Size([1, 12, 2])
    
    import time 
    start = time.time()
    output = rnnTrajectory(input)
    end = time.time()
    print("time ", 1 / (end - start))
    print(output)
    print(output.size()) # torch.Size([2, 1, 7])
   
    start = time.time()
    output = lstmTrajectory(input)
    end = time.time()
    print("time ", 1 / (end - start))
    print(output)
    print(output.size())

    start = time.time()
    output = gruTrajectory(input)
    end = time.time()
    print("time ", 1 / (end - start))
    print(output)
    print(output.size())
        