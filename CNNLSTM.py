import re
from collections import OrderedDict
from typing import Callable, Sequence, Type, Union

import monai

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class CNNLSTM(torch.nn.Module):
    def __init__(self, encoder_name='efficientnet-b0' ,spatial_dims=1, in_channels=1, num_classes=1, encoderStateDictPath=None):
        super(CNNLSTM, self).__init__()        
        
        """
        input shape : for spatial dims=1, x should be ( Batch x Channel x Feature x Depth )
                      for spatial dims=2, x should be ( Batch x Channel x Height x Weight x Depth )
                      for spatial dims=3, x should be ( Batch x Channel x Height x Weight x Slices x Depth )
        
        spatial_dims : type of convolution, [conv1d, conv2d, conv3d]
        in_channels : input channels
        num_classes : final output channels after LSTM
        encoderStateDictPath : If you have pretrained encoder state dict, enter the path of saved state dict, load encoder will be freezed!
        """
        
        # Feature Extraction
        self.encoder = monai.networks.nets.EfficientNetBNFeatures(encoder_name, 
                                                                  progress=True, 
                                                                  spatial_dims=spatial_dims, 
                                                                  in_channels=in_channels,
                                                                  norm= 'batch',
                                                                  num_classes=1000)
        if encoderStateDictPath:
            weight = torch.load(encoderStateDictPath)
            self.encoder.load_state_dict(weight,strict=False)
            self.encoder.eval()
            
        if spatial_dims==1:
            self.pool    = torch.nn.AdaptiveAvgPool1d(1)
            x = torch.rand(2,in_channels,64)
            feature = self.encoder(x)
            num_channelLastFeature = feature[-1].shape[1] # get num_channelLastFeature, this is also input size of LSTM            
        elif spatial_dims==2:
            self.pool    = torch.nn.AdaptiveAvgPool2d(1)
            x = torch.rand(2,in_channels,64,64)
            feature = self.encoder(x)
            num_channelLastFeature = feature[-1].shape[1] # get num_channelLastFeature, this is also input size of LSTM
        elif spatial_dims==3:
            self.pool    = torch.nn.AdaptiveAvgPool3d(1)
            x = torch.rand(2,in_channels,64,64,64)
            feature = self.encoder(x)
            num_channelLastFeature = feature[-1].shape[1] # get num_channelLastFeature, this is also input size of LSTM
            
        # LSTM Classifier
        self.LSTM    = nn.LSTM(input_size=num_channelLastFeature, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.fc      = nn.Linear(512*2, 512, True)
        self.relu    = nn.ReLU(True)
        self.drop    = nn.Dropout(p=0.2)
        
        # Head
        self.head    = nn.Linear(512, num_classes, True)       

    def forward(self, x, depths): 
        """
        x is input with (Batch x Sequence x Feature), depths is list with Sequence of each batch
        """
        
        # stacking Features 
        encoder_embed_seq = []
        for i in range(x.shape[-1]):
            out = self.encoder(x[..., i])[-1]    
            out = self.pool(out)
            out = out.view(out.shape[0], -1)
            encoder_embed_seq.append(out)   
            
        stacked_feat = torch.stack(encoder_embed_seq, dim=1)
        
        # Classifier, Input = (Batch, Seq, Feat)
        self.LSTM.flatten_parameters()  
        x_packed = pack_padded_sequence(stacked_feat, depths, batch_first=True, enforce_sorted=False)
#         print(x_packed) # Check
        RNN_out, (h_n, h_c) = self.LSTM(x_packed, None)    
        fc_input = torch.cat([h_n[-1], h_n[-2]], dim=-1) # Due to the Bi-directional
        x = self.fc(fc_input)
        x = self.relu(x)  
        x = self.drop(x)  
        
        # Head
        x = self.head(x)     
        return x

###########################################################################      
# To use our model, spatial_dims=1, in_channels=1, num_classes=1
model = CNNLSTM(spatial_dims=1, in_channels=1, num_classes=1).cuda()

x = torch.rand(6,1,2048,8)
yhat = model(x.cuda(),[7,6,4,6,5,3])
print(x.shape, yhat.shape)

###########################################################################
# To use our model, spatial_dims=1, in_channels=1, num_classes=2
model = CNNLSTM(spatial_dims=1, in_channels=1, num_classes=2).cuda()

x = torch.rand(6,1,2048,8)
yhat = model(x.cuda(),[7,6,4,6,5,3])
print(x.shape, yhat.shape)

###########################################################################
# To use our model, spatial_dims=2, in_channels=1, num_classes=1
model = CNNLSTM(spatial_dims=2, in_channels=1, num_classes=1).cuda()

x = torch.rand(6,1,64,64,8)
yhat = model(x.cuda(),[7,6,4,6,5,3])
print(x.shape, yhat.shape)

###########################################################################
# To use our model, spatial_dims=3, in_channels=1, num_classes=1
model = CNNLSTM(spatial_dims=3, in_channels=1, num_classes=1).cuda()

x = torch.rand(6,1,64,64,64,8)
yhat = model(x.cuda(),[7,6,4,6,5,3])
print(x.shape, yhat.shape)
