from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class highwayNet(nn.Module):  # 从pytorch中继承类

    ## Initialization
    def __init__(self,args):
        super(highwayNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.use_maneuvers = args['use_maneuvers']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size']
        self.input_embedding_size = args['input_embedding_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth
        # size(soc_embedding_size)=80
        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)  # input:2, output:32  全连接层

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)  # input:32, 隐藏层特征维度：64， LSTM个数：1  LSTM层

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)  # input:64, output：32  全连接层

        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)  # in_channels:64, out_channels:64, kernel_size=3*3  卷积层
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1))  # in_channels: 64, out_channels: 16, kernel_size=3*1  卷积层
        self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0))  # kernel size: 2*1  池化层

        # FC social pooling layer (for comparison):
        # self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)

        # Decoder LSTM
        if self.use_maneuvers:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
            #  input: 80+32+3+2, output: 128
        else:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)
            #  input: 80+32, output:128

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size,5)  # input: 128, output: 5
        self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)  # input:80+32, output: 3
        self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)  # input:80+32, output: 2

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.init_parameters()


    ## Forward Pass
    def forward(self,hist,nbrs,masks,lat_enc,lon_enc):

        ## Forward pass hist:
        _,(hist_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))  # in_emb: 2->32, hist.shape = [16, 128, 2]
        # hist_enc.shape = [1, 128, 64]
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))
        # hist_enc.shape = [128, 32]
        
        ## Forward pass nbrs
        _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        # nbrs_enc.shape= [1, 857, 64]
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])  # reshape
        # nbrs_enc.shape = [857, 64]

        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float()
        # soc_enc.shape = [128, 3, 13, 64]
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)  # 用于替换的值
        # soc_enc.shape = [128, 3, 13, 64]
        soc_enc = soc_enc.permute(0,3,2,1)  # 维度的顺序转换
        # soc_enc.shape = [128, 64, 13, 3]
        
        ## Apply convolutional social pooling:
        soc_enc = self.soc_maxpool(self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc)))))
        # soc_enc.shape = [128, 16, 5, 1]
        
        soc_enc = soc_enc.view(-1,self.soc_embedding_size)
        # soc_enc.shape = [128, 80]

        ## Apply fc soc pooling
        # soc_enc = soc_enc.contiguous()
        # soc_enc = soc_enc.view(-1, self.soc_conv_depth * self.grid_size[0] * self.grid_size[1])
        # soc_enc = self.leaky_relu(self.soc_fc(soc_enc))

        ## Concatenate encodings:
        enc = torch.cat((soc_enc,hist_enc),1)  # 将两个tensor按列拼接在一起
        # enc.shape = [128, 112]


        if self.use_maneuvers:
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc))  # 辨识当前侧向行为, 三个类别的概率值
            # lat_pred.shape = [128, 3]
            
            lon_pred = self.softmax(self.op_lon(enc))  # 分类出当前纵向行为, 两个类别的概率值
            # lon_pred = [128, 2]

            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_enc, lon_enc), 1)
                # enc.shape = [128 117]
                fut_pred = self.decode(enc)
                # fut_pred.shape = [25, 128, 5]
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(enc)
            return fut_pred


    def decode(self,enc):  # enc.shape = [128, 117]
        enc = enc.repeat(self.out_length, 1, 1)  # enc.shape = [25, 128, 117]
        h_dec, _ = self.dec_lstm(enc)  # h_dec.shape = [25, 128, 128]
        h_dec = h_dec.permute(1, 0, 2)  # h_dec.shape = [128, 25, 128]
        fut_pred = self.op(h_dec)  # fut_pred.shape = [128, 25, 5]
        fut_pred = fut_pred.permute(1, 0, 2)  # fut_pred = [25, 128, 5]
        fut_pred = outputActivation(fut_pred)  # fut_pred = [25, 128, 5]
        return fut_pred
        
        
    def init_parameters(self):
        nn.init.zeros_(self.ip_emb.bias)
        nn.init.kaiming_normal_(self.ip_emb.weight)
        nn.init.zeros_(self.dyn_emb.bias)
        nn.init.kaiming_normal_(self.dyn_emb.weight)
        nn.init.zeros_(self.soc_conv.bias)
        nn.init.kaiming_normal_(self.soc_conv.weight)
        nn.init.zeros_(self.conv_3x1.bias)
        nn.init.kaiming_normal_(self.conv_3x1.weight)
        nn.init.zeros_(self.op.bias)
        nn.init.kaiming_normal_(self.op.weight)
        nn.init.zeros_(self.op_lat.bias)
        nn.init.kaiming_normal_(self.op_lat.weight)
        nn.init.zeros_(self.op_lon.bias)
        nn.init.kaiming_normal_(self.op_lon.weight)

        for name, param in self.enc_lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

        for name, param in self.dec_lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)





