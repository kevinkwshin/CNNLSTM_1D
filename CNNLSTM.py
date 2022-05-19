from torch import nn
from torch.nn import functional as F

import re
from collections import OrderedDict
from typing import Callable, Sequence, Type, Union

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

import monai
from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.module import look_up_option
   
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from monai.networks.nets.densenet import _DenseLayer, _DenseBlock, _Transition

class DenseNetFeature(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        dropout_prob: float = 0.0,
    ) -> None:

        super().__init__()

        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        avg_pool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )
        self.conv0 = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )
        
        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            if i==0:
                self.layer1 = block
            if i==1:
                self.layer2 = block
            if i==2:
                self.layer3 = block
            if i==3:
                self.layer4 = block
                
            in_channels += num_layers * growth_rate
            if i==3:
                self.norm5 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)

            else:
                _out_channels = in_channels // 2
                trans = _Transition(
                    spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm
                )
                # self.features.add_module(f"transition{i + 1}", trans)
                
                if i==0:
                    self.transition1 = trans
                if i==1:
                    self.transition2 = trans
                if i==2:
                    self.transition3 = trans
                in_channels = _out_channels

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x1 = self.layer1(x)
        x = self.transition1(x1)
        x2 = self.layer2(x)
        x = self.transition2(x2)
        x3 = self.layer3(x)
        x = self.transition3(x3)
        x = self.layer4(x)
        x4 = self.norm5(x)
        return x1,x2,x3,x4

class CNNLSTM_1D(torch.nn.Module):
    def __init__(self, spatial_dims=1, in_channels=1, num_classes=1, encoderStateDictPath=None):
        super(CNNLSTM_1D, self).__init__()        
        
        # Feature Extraction
        
        self.encoder = DenseNetFeature(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=1000, norm=('group',{'num_groups':32}))
        self.encoder = monai.networks.nets.EfficientNetBNFeatures('efficientnet-b1', 
                                                                  progress=True, 
                                                                  spatial_dims=spatial_dims, 
                                                                  in_channels=in_channels,
                                                                  norm= 'batch',
                                                                  num_classes=1000)
        if encoderStateDictPath:
            weight = torch.load(encoderStateDictPath)
            self.encoder.load_state_dict(weight,strict=False)
            self.encoder.eval()
            
        x = torch.rand(2,in_channels,2048)
        feature = self.encoder(x)
        num_channelLastFeature = feature[-1].shape[1] # get num_channelLastFeature, this is also input size of LSTM
        self.pool    = torch.nn.AdaptiveAvgPool1d(1)
        
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
model = CNNLSTM_1D(spatial_dims=1, in_channels=1, num_classes=1).cuda()

x = torch.rand(6,1,2048,8)
yhat = model(x.cuda(),[7,6,4,6,5,3])
print(x.shape, yhat.shape)

###########################################################################
# To use our model, spatial_dims=1, in_channels=1, num_classes=2
model = CNNLSTM_1D(spatial_dims=1, in_channels=1, num_classes=2).cuda()

x = torch.rand(6,1,2048,8)
yhat = model(x.cuda(),[7,6,4,6,5,3])
print(x.shape, yhat.shape)
