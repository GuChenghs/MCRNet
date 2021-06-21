import torch.nn as nn
import torch
import torch.nn.functional as F
from models.dropblock import DropBlock
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import transforms
import torch.optim as optim

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding,input in_planes channels ,out out_planes channels"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Encoder(torch.nn.Module): #Encoder
    def __init__(self,in_plane,out_plane):
        super(Encoder,self).__init__()
        self.layer1 = torch.nn.Conv2d(in_plane,out_plane,kernel_size= 3,stride= 1,padding=1) #28*28*1--->14*14*out_plane

    def forward(self, x):

        return F.relu(self.layer1(x))


class Decoder(torch.nn.Module): #Decoder
    def __init__(self,z_plane,Max_plane,in_plane):
        super(Decoder,self).__init__()
        self.layer1 = torch.nn.ConvTranspose2d(z_plane,Max_plane,kernel_size= 3,stride= 1)
        self.layer2 = torch.nn.ConvTranspose2d(Max_plane,in_plane,kernel_size= 3,stride=1)
        #self.layer3 = torch.nn.ConvTranspose2d(in_plane,out_plane,kernel_size=3,stride=1)

    def forward(self, x):
        #x = F.relu(self.layer1(x))
        #x = F.relu(self.layer2(x))
        return F.relu(self.layer1(x))

class VAE(torch.nn.Module):
    #latent_dim = 8

    def __init__(self, encoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.relu = nn.LeakyReLU(0.1)
        #self.decoder = decoder
        # self._enc_mu = torch.nn.Linear(100, 8)
        # self._enc_log_sigma = torch.nn.Linear(100, 8)
        self._enc_mu = torch.nn.Conv2d(32,64,kernel_size= 3,stride= 1,padding=1)
        self._enc_log_sigma = torch.nn.Conv2d(32,64,kernel_size= 3,stride= 1,padding=1)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)   #32*6*6
        log_sigma = self._enc_log_sigma(h_enc)  #32*6*6
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()     #sample from normal distribution with size which could be none int or matrix

        self.z_mean = mu
        self.z_sigma = sigma
        std_z = std_z.cuda()

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick，without daoshu  32*8

    def forward(self, state):
        h_enc = self.encoder(state)
        h_enc = h_enc.cuda()
        z = self._sample_latent(h_enc)
        z = z.cuda()
        return z

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                if (feat_size - self.block_size + 1)!=0:
                    gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                else:
                    gamma = (1 - keep_rate) / self.block_size**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class GNN_Block(torch.nn.Module):
    #latent_dim = 8

    def __init__(self, encoder):
        super(GNN_Block, self).__init__()
        self.encoder = encoder
        self.relu = nn.LeakyReLU(0.1)
        #self.decoder = decoder
        # self._enc_mu = torch.nn.Linear(100, 8)
        # self._enc_log_sigma = torch.nn.Linear(100, 8)
        self._enc_mu = torch.nn.Conv2d(32,64,kernel_size= 3,stride= 1,padding=1)
        self._enc_log_sigma = torch.nn.Conv2d(32,64,kernel_size= 3,stride= 1,padding=1)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)   #32*6*6
        log_sigma = self._enc_log_sigma(h_enc)  #32*6*6
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()     #sample from normal distribution with size which could be none int or matrix

        self.z_mean = mu
        self.z_sigma = sigma
        std_z = std_z.cuda()

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick，without daoshu  32*8

    def forward(self, state):
        h_enc = self.encoder(state)
        h_enc = h_enc.cuda()
        z = self._sample_latent(h_enc)
        z = z.cuda()
        return z
    
class ResNet(nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 64

        super(ResNet, self).__init__()

        self.encoder = Encoder(3,32)

        self.vae = VAE(self.encoder)

        self.layer1 = self._make_layer(block, 128, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    #insert variation block
    # def variational_embedding(self,vae):
    #     encoder = Encoder(3, 64, 160)
    #     decoder = Decoder(320, 160, 64, 3)
    #     vae = VAE(encoder, decoder)
    #     x = x.type(torch.FloatTensor)
    #     x = vae(x)
    #     return x

    def forward(self, x):
        x = self.vae(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


 

def resnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model
