import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from ..registry import BACKBONES
from ..weight_init import weight_init_
import paddlenlp


# tools and graph
def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
         - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak


def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A


num_node = 25
self_link = [(i, i) for i in range(num_node)]

# ntu-rgb-d
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]

# fsd competition
# inward_ori_index = [(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
#                     (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
#                     (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
#                     (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
#                     (21, 14), (19, 14), (20, 19)]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph():
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


# CTR-GCN
class TemporalConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2D(branch_channels),
                nn.Swish(),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2D(branch_channels),
            nn.Swish(),
            nn.MaxPool2D(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2D(branch_channels) 
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2D(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)


    def forward(self, x):

        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = paddle.concat(branch_outs, axis=1)
        out += res
        return out


class CTRGC(nn.Layer):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9 or in_channels == 2:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction

        self.conv1 = nn.Conv2D(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2D(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2D(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2D(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()


    def forward(self, x, A=None, alpha=1):

        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        x3 = self.conv3(x)

        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)

        x1 = paddlenlp.ops.einsum('ncuv,nctv->nctu', x1, x3)

        return x1


class unit_tcn(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2D(out_channels)


    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Layer):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.LayerList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2D(in_channels, out_channels, 1),
                    nn.BatchNorm2D(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        if self.adaptive:
            self.PA = paddle.to_tensor(A.astype(np.float32), stop_gradient=False)
        else:
            self.A = paddle.to_tensor(A.astype(np.float32), stop_gradient=True)

        self.alpha = paddle.to_tensor(np.zeros(1), stop_gradient=False)

        self.bn = nn.BatchNorm2D(out_channels)
        self.soft = nn.Softmax(-2)
        self.swish = nn.Swish()


    def forward(self, x):

        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A  

        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.swish(y)
        return y


class TCN_GCN_unit(nn.Layer):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.swish = nn.Swish()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.swish(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


@BACKBONES.register()
class CTRGCN(nn.Layer):
    def __init__(self, num_class=30, num_point=25, num_person=1, graph=None, graph_args=dict(), in_channels=2,
                 drop_out=0, adaptive=True, data_bn=True, ):
        super(CTRGCN, self).__init__()

        self.data_bn = data_bn
        # load graph
        self.graph = Graph()
        A = self.graph.A  # 3,25,25
        self.num_class = num_class
        self.num_point = num_point

        self.data_bn = nn.BatchNorm1D(num_person * in_channels * num_point)

        base_channel = 64
        self.st_gcn_networks = nn.LayerList((
            TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive),
            TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive),
            TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive),
            TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive),
            TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive),
            TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive),
            TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive),
            TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive),
            TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive),
            TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive),
        ))
        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

    def init_weights(self):
        """Initiate the parameters.
        """
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                weight_init_(layer, 'Normal', mean=0.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm2D):
                weight_init_(layer, 'Normal', mean=1.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm1D):
                weight_init_(layer, 'Normal', mean=1.0, std=0.02)

    def forward(self, x):

        N, C, T, V, M = x.shape  # [32, 2, 350, 25, 1]

        x = x.transpose((0, 4, 3, 1, 2))  # N, M, V, C, T

        x = x.reshape((N * M, V * C, T))

        if self.data_bn:
            x.stop_gradient = False
        x = self.data_bn(x)

        x = x.reshape((N, M, V, C, T))

        x = x.transpose((0, 1, 3, 4, 2))  # N, M, C, T, V

        x = x.reshape((N * M, C, T, V))

        for gcn in self.st_gcn_networks:
            x = gcn(x)

        x = self.pool(x)  # NM,C,T,V --> NM,C,1,1
        C = x.shape[1]
        x = paddle.reshape(x, (N, M, C, 1, 1)).mean(axis=1)  # N,C,1,1

        return x