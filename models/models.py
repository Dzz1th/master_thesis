import torch
import torch.nn as nn

class TripletLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_layers=1):
        super(TripletLinearClassifier, self).__init__()

        self.layers = nn.ModuleList()
        current_dim = input_dim

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(current_dim, current_dim//2))
            self.layers.append(nn.Sigmoid())
            current_dim = current_dim//2
        
        self.layers.append(nn.Linear(current_dim, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x.squeeze(-1)  # Output a scalar for each input
    


    

class TripletLinearClassifier2Layers(nn.Module):
    def __init__(self, input_dim):
        super(TripletLinearClassifier2Layers, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x.squeeze(-1)  # Output a scalar for each input