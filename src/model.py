import torch
import timm
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torch.cuda.amp import autocast

sigmoid = nn.Sigmoid()

class MelanomaNet(nn.Module):
    """
    Model architecture for training
    """
    def __init__(self,out_dim=9,network_1="tf_efficientnet_b0_ns",network_2="seresnet101"):
        """
        Construct NeuralNetwork object and initialize member variables
        """
        super(MelanomaNet, self).__init__()
        
        self.modelA = timm.create_model(network_1,pretrained=True)
        self.modelB = timm.create_model(network_2,pretrained=True)
        num_features_A = self.modelA.classifier.in_features
        num_features_B = self.modelB.fc.in_features
        self.modelA.classifier = nn.Identity()
        self.modelB.fc = nn.Identity()
        self.classifier = nn.Linear(num_features_A+num_features_B, out_dim)
    def forward(self,x):
        x1 = self.modelA(x.clone())
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
        
class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class MetaMelanoma(nn.Module):
    def __init__(self,out_dim=9,n_meta_features=0,n_meta_dim=[512, 128],network='efficientnet_b0'):
            super(MetaMelanoma,self).__init__()
            self.enet = timm.create_model(network,pretrained=True)
            self.n_meta_features = n_meta_features
            self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)])
            in_ch = self.enet.classifier.in_features
            if n_meta_features > 0:
                self.meta = nn.Sequential(
                    nn.Linear(n_meta_features, n_meta_dim[0]),
                    nn.BatchNorm1d(n_meta_dim[0]),
                    Swish_Module(),
                    nn.Dropout(p=0.3),
                    nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                    nn.BatchNorm1d(n_meta_dim[1]),
                    Swish_Module(),
                )
                in_ch += n_meta_dim[1]
            self.myfc = nn.Linear(in_ch, out_dim)
            self.enet.classifier = nn.Identity()
    def extract(self, x):
        x = self.enet(x)
        return x
    def forward(self, x,x_meta):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out

class BaseNetwork(nn.Module):
    def __init__(self,network):
        super(BaseNetwork, self).__init__()
        self.pretrained_block = timm.create_model(network,pretrained=True,num_classes=9)
    def forward(self,x):
        x = self.pretrained_block(x)
        return x


if __name__ == '__main__':
    trial_model = BaseNetwork(network='resnet18')
    print(trial_model)
