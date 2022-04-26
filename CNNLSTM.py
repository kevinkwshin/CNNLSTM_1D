from torch import nn
from torch.nn import functional as F

import re
from collections import OrderedDict
from typing import Callable, Sequence, Type, Union

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.module import look_up_option

class _DenseLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            growth_rate: how many filters to add each layer (k in paper).
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.layers.add_module("relu1", get_act_layer(name=act))
        self.layers.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels))
        self.layers.add_module("relu2", get_act_layer(name=act))
        self.layers.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            layers: number of layers in the block.
            in_channels: number of the input channel.
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            growth_rate: how many filters to add each layer (k in paper).
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob, act=act, norm=norm)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            out_channels: number of the output classes.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.add_module("relu", get_act_layer(name=act))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


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
                
            # self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            # if i == len(block_config) - 1:
            #     self.features.add_module(
            #         "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
            #     )
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

        # # pooling and classification
        # self.class_layers = nn.Sequential(
        #     OrderedDict(
        #         [
        #             ("relu", get_act_layer(name=act)),
        #             ("pool", avg_pool_type(1)),
        #             ("flatten", nn.Flatten(1)),
        #             ("out", nn.Linear(in_channels, out_channels)),
        #         ]
        #     )
        # )

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.features(x)
        # x = self.class_layers(x)
        
        x = self.conv0(x)
        x1 = self.layer1(x)
        x = self.transition1(x1)
        x2 = self.layer2(x)
        x = self.transition2(x2)
        x3 = self.layer3(x)
        x = self.transition3(x3)
        x = self.layer4(x)
        x4 = self.norm5(x)

        # return x1,x2,x3,x4
        return x4

class CNNLSTM(nn.Module):
    def __init__(self,cnn, num_classes = 1, embed_size=1024, LSTM_UNITS=64, DO = 0.3):
        super(CNNLSTM, self).__init__()
        # self.cnn = cnn.module
        self.cnn = cnn
        self.cnn.eval().cuda()
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.linear_pe = nn.Linear(LSTM_UNITS*2, 1)

    def forward(self, x, lengths=None):
        embedding 
        with torch.no_grad():
            
            # embedding = self.cnn.extract_features(x)
            embedding = self.cnn(x)
            embedding = self.avgpool(embedding)
            b,f,_ = embedding.shape
            embedding = embedding.reshape(1,b,f)
        print('hehe')
        self.lstm1.flatten_parameters()
        h_lstm1, _ = self.lstm1(embedding)
        self.lstm2.flatten_parameters()
        h_lstm2, _ = self.lstm2(h_lstm1)
        print(h_lstm2.shape)
        
        h_conc_linear1  = F.relu(self.linear1(h_lstm1))
        h_conc_linear2  = F.relu(self.linear2(h_lstm2))
        
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2

        output = self.linear_pe(hidden)
        return output
    
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class CNNLSTM(torch.nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()        
        
        # Slicewise Feature Extract
        # self.encoder = net.encoder
        self.encoder = net
        self.pool    = torch.nn.AdaptiveAvgPool1d(1)
        
        # 3D Classifier - ResNet50 based
        self.LSTM    = nn.LSTM(input_size=1024, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.fc      = nn.Linear(512*2, 512, True)
        self.relu    = nn.ReLU(True)
        self.drop    = nn.Dropout(p=0.5)
        
        # Head
        self.head    = nn.Linear(512, 1, True)       

    def forward(self, x, depths):
        
        # D stacking Features  
        encoder_embed_seq = []
        for i in range(x.shape[-1]):
            out = self.encoder(x[..., i])#.round()        
            out = self.pool(out)
            out = out.view(out.shape[0], -1)
            encoder_embed_seq.append(out)   
            
        stacked_feat = torch.stack(encoder_embed_seq, dim=1)
        # 3D Classifier, Input = (Batch, Seq, Feat)
        self.LSTM.flatten_parameters()  
        x_packed = pack_padded_sequence(stacked_feat, depths, batch_first=True, enforce_sorted=False)
        print(x_packed)
        RNN_out, (h_n, h_c) = self.LSTM(x_packed, None)    
        fc_input = torch.cat([h_n[-1], h_n[-2]], dim=-1) # Due to the Bi-directional
        x = self.fc(fc_input)
        x = self.relu(x)  
        x = self.drop(x)  
        
        # Head
        x = self.head(x)     

        return x
    
net = DenseNetFeature(spatial_dims=1, in_channels=1, out_channels=1, norm=('group',{'num_groups':32}))
model = CNNLSTM().cuda()

x = torch.rand(6,1,2048,8)
yhat = model(x.cuda(),[7,6,4,6,5,3])
print(x.shape, yhat.shape)