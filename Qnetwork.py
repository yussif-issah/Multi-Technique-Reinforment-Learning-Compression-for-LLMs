import torch.nn as nn

class Qnetwork(nn.Module):
    def __init__(self, obs_shape:int, num_actions:int, hidden_size:list[int] = [1024,512,256,128,64,32,16]):
        super().__init__()
        in_features = [obs_shape] + hidden_size
        out_features = hidden_size + [num_actions]

        layers = []
        for i, (in_f, out_f) in enumerate(zip(in_features, out_features)):
            layers.append(nn.Linear(in_f, out_f))
            if i < len(in_features) - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    


        