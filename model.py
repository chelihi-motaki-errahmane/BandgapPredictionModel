#Creating the Model
import torch
import torch.nn as nn
class BandgapModel(nn.Module):
    def __init__(self, input_dim):
        super(BandgapModel, self).__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Residual Block
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        
        # Output Layers
        self.fc4 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        
        # Residual connection
        residual = x
        out = self.relu(self.fc2(x))
        out = self.fc3(out)
        x = self.relu(out + residual) 
        
        x = self.dropout(self.relu(self.fc4(x)))
        return self.output(x)