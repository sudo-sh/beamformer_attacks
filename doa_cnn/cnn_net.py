import torch
import torch.nn as nn
import os

import utils

        
class ConvNet2D(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.num_class = param.num_class
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(param.dropout))
        

        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(param.dropout))

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(param.dropout))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(param.dropout))

       
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*11*11,4096), 
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(param.dropout))
        
        self.layer6 = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(4096,2048), 
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(param.dropout))

        self.layer7 = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(2048,1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(param.dropout))
        
        
        self.layer8 = nn.Sequential(
            nn.Linear(1024,self.num_class),
            nn.Sigmoid()
            )

    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.shape)
        out = self.layer5(out)
        # print(out.shape)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        # print(out.shape)
        return out
    
if __name__=='__main__':

    json_path = os.path.join("model", 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    model = ConvNet2D(params)
    data = torch.randn((64, 3, 16, 16))
    out = model(data)
