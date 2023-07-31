import torch
from torch import nn
from lavis.models.gram_models import neuralnet

class EmotionClassifier(nn.Module):
    def __init__(self,name=None,drop=None,freeze=None,mlp=None,dropout=None,activations=None,factors=None,level=None,d_model=None,nhead=None,dim_feedforward=None,num_layers=None):
        super().__init__()
        self.net=neuralnet.BaseNet(name=name,drop=drop,freeze_level=freeze)
        self.fc_layers=neuralnet.MLP(mlp,dropout,activations)
        self.num_features=512
        #self.logits=nn.Sequential(self.net, self.fc)#neuralnet.BaseNet(name=name,drop=1),neuralnet.MLP(mlp,dropout,activations))

    def forward(self,x):
        x=self.net(x[:,:3,:,:])#color only feed
        return self.fc_layers(x)
    
    def collect_feat(self,x):
        x=self.net(x[:,:3,:,:])
        return x

    def get_num_layer(self):
        return len(list(self.net.children()))
        
