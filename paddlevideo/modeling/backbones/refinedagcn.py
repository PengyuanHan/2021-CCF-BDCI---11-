import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..registry import BACKBONES

class GCN(nn.Layer):
    def __init__(self, in_channels, out_channels, vertex_nums=25, stride=1, Attention=True):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=in_channels,
                               out_channels=3 * out_channels,
                               kernel_size=1,
                               stride=1)
        self.conv2 = nn.Conv2D(in_channels=vertex_nums * 3,
                               out_channels=vertex_nums,
                               kernel_size=1)

        self.soft = nn.Softmax(-2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.conv_concat = nn.Conv2D(3*out_channels, out_channels, 1)

        if Attention:
            # temporal attention
            self.conv_TA = nn.Conv1D(out_channels, 1, 9, padding=4)


            # s attention
            self.conv_SA = nn.Conv1D(out_channels, 1, 25, padding=24 // 2, 
                                                    weight_attr=nn.initializer.Constant(value=0.0))
            nn.initializer.XavierNormal(self.conv_SA.weight)
            

            #channel attention
            self.fc1 = nn.Linear(out_channels, out_channels // 2, 
                                                    bias_attr=nn.initializer.Constant(value=0.0))
            nn.initializer.KaimingNormal(self.fc1.weight)
            self.fc2 = nn.Linear(out_channels // 2, out_channels, 
                                                    weight_attr=nn.initializer.Constant(value=0.0), 
                                                    bias_attr=nn.initializer.Constant(value=0.0))
        self.Attention = Attention
            

    def forward(self, x):
        """自适应图卷积的实现。"""
        # x --- N,C,T,V
        y = self.conv1(x)  # N,3C,T,V
        N, C, T, V = y.shape
        y = paddle.reshape(y, [N, C // 3, 3, T, V])  # N,C,3,T,V
        y = paddle.transpose(y, perm=[0, 1, 2, 4, 3])  # N,C,3,V,T
        y = paddle.reshape(y, [N, C // 3, 3 * V, T])  # N,C,3V,T
        y = paddle.transpose(y, perm=[0, 2, 1, 3])  # N,3V,C,T
        y = self.conv2(y)  # N,V,C,T
        y = paddle.transpose(y, perm=[0, 2, 3, 1])  # N,C,T,V

        if self.Attention:
            # spatial attetnion
            SA = y.mean(-2)
            SA_1 = self.sigmoid(self.conv_SA(SA))
            y1 = y * (paddle.unsqueeze(SA_1, -2))

            # temporal attention
            TA = y.mean(-1)
            TA_1 = self.sigmoid(self.conv_SA(TA)) 
            y2 = y * (paddle.unsqueeze(TA_1, -1))

            # channel attention
            CA = y.mean(-1).mean(-1)
            CA_1 = self.relu(self.fc1(CA))
            CA_2 = self.sigmoid(self.fc2(CA_1))
            y3 = y * (paddle.unsqueeze(paddle.unsqueeze(CA_2, -1), -1))

            y = paddle.concat([y1, y2, y3], axis=1)
            y = self.conv_concat(y)
            #y += self.down(x)
            y = self.relu(y)

        return y

"""修改Block模块：2s-AGCN。"""
class Block(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 Attention=True,
                 vertex_nums=25,
                 temporal_size=9,
                 stride=1,
                 residual=True):
        super(Block, self).__init__()
        self.residual = residual
        self.out_channels = out_channels

        self.bn_res = nn.BatchNorm2D(out_channels)
        self.conv_res = nn.Conv2D(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=(stride, 1))
        self.gcn = GCN(in_channels=in_channels,
                       out_channels=out_channels,
                       vertex_nums=vertex_nums, Attention=Attention)
        self.tcn = nn.Sequential(
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(temporal_size, 1),
                      padding=((temporal_size - 1) // 2, 0),
                      stride=(stride, 1)),
            nn.BatchNorm2D(out_channels),
        )

    def forward(self, x):
        if self.residual:
            y = self.conv_res(x)
            y = self.bn_res(y)
        x = self.gcn(x)
        x = self.tcn(x)
        out = x + y if self.residual else x
        out = F.relu(out)
        return out

@BACKBONES.register()
class REFINEDAGCN(nn.Layer):
    def __init__(self, in_channels=2, **kwargs):
        super(REFINEDAGCN, self).__init__()


        self.agcn = nn.Sequential(
            Block(in_channels=in_channels, out_channels=64, Attention=True, residual=False, **kwargs), 
            Block(in_channels=64, out_channels=64, Attention=True, **kwargs),
            Block(in_channels=64, out_channels=64, Attention=True, **kwargs),
            Block(in_channels=64, out_channels=64, Attention=True, **kwargs),
            Block(in_channels=64, out_channels=128, Attention=True, stride=2, **kwargs),
            Block(in_channels=128, out_channels=128, Attention=True, **kwargs),
            Block(in_channels=128, out_channels=128, Attention=True, **kwargs),
            Block(in_channels=128, out_channels=256, Attention=True, stride=2, **kwargs),
            Block(in_channels=256, out_channels=256, Attention=True, **kwargs),
            Block(in_channels=256, out_channels=256, Attention=True, **kwargs)
            )

        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.shape

        x = x.transpose((0, 4, 1, 2, 3))  # N, M, C, T, V
        x = x.reshape((N * M, C, T, V))

        x = self.agcn(x)

        x = self.pool(x)  # NM,C,T,V --> NM,C,1,1
        C = x.shape[1]
        x = paddle.reshape(x, (N, M, C, 1, 1)).mean(axis=1)  # N,C,1,1

        return x