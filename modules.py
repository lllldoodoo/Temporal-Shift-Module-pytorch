import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import math

def tsm(tensor, duration, version='zero'):
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])
    # tensor [N, T, C, H, W]
    pre_tensor, post_tensor, peri_tensor = tensor.split([size[1] // 4,
                                                         size[1] // 4,
                                                         size[1] // 2], dim=2)
    if version == 'zero':
        pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]
        post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]
    elif version == 'circulant':
        pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],
                                 pre_tensor [:,   :-1, ...]), dim=1)
        post_tensor = torch.cat((post_tensor[:,  1:  , ...],
                                 post_tensor[:,   :1 , ...]), dim=1)
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))
    return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(size)

class TSM(nn.Module):

    def __init__(self, num_segments, version='zero'):
        super(TSM, self).__init__()
        self.num_segments = num_segments
        self.version = version

    def forward(self, x):

        return tsm(x, self.num_segments, self.version)
    
class Self_Attention(nn.Module):

    def __init__(self, num_segments, dim_in, dim_k=64):
        super(Self_Attention, self).__init__()
        self.num_segments = num_segments
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(dim_in, dim_k)

    def forward(self, x):
        # x: [N * T, C, H, W]
        size = x.size()
        x = x.view((-1, size[1], size[2]*size[3]))
        # x: [N * T, C, H * W]
        Q = x.max(dim=-1)[0]
        # Q: [N * T, C]
        Q = self.linear(Q)
        # Q: [N * T, k]
        Q = Q.view((-1, self.num_segments, self.dim_k))
        # Q: [N, T, k]
        A = torch.matmul(Q, Q.transpose(1, 2))
        # A: [N, T, T]
        A = A/ math.sqrt(self.dim_k)
        A = self.softmax(A)
        A = self.dropout(A)

        x = x.view((-1, self.num_segments, size[1]*size[2]*size[3]))
        # x: [N, T, C * H * W]
        output = torch.matmul(A, x)
        # out: [N, T, C * H * W]
        output = output.view(size)

        return output

    
class ConcatShift(nn.Module):
    def __init__(self, num_segments, reverse=False):
        super(ConcatShift, self).__init__()
        self.num_segments = num_segments
        self.reverse = reverse
        
    def forward(self,x):
        size = x.size()
        # (N*T, C, H, W)
        x = x.view((-1, self.num_segments) + size[1:])
        # (N, T, C, H, W)
        if not self.reverse:
            x_shift = torch.zeros_like(x)
            x_shift[:, 1:, ...] = x[:, :-1, ...]
            x_concat = torch.cat((x_shift, x), dim=2)
        else:
            x_shift = torch.zeros_like(x)
            x_shift[:, 1:, ...] = x[:, :-1, ...]
            x_reverse_shift = torch.zeros_like(x)
            x_reverse_shift[:, :-1, ...] = x[:, 1:, ...]
            x_concat = torch.cat((x_shift, x, x_reverse_shift), dim=2)
            
        return x_concat.view((size[0], -1) + size[2:])
    
    
    
    

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
    
        
class TemporalChannelGate(nn.Module):
    def __init__(self, num_segments, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], dim_k=64):
        super(TemporalChannelGate, self).__init__()
        self.num_segments = num_segments
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.dim_k = dim_k
        self.linear = nn.Linear(gate_channels*2, dim_k)
    def forward(self, x):
        channel_att_sum = None
        pool_concat = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
                
        pool = torch.cat((avg_pool, max_pool), dim=1).squeeze()
        pool = self.linear(pool)
        pool = pool.view((pool.size(0), self.num_segments, -1))
        

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class TemporalGate(nn.Module):
    def __init__(self, num_segments):
        super(TemporalGate, self).__init__()
        self.num_segments = num_segments
        self.gate = ChannelGate(num_segments, 2, ['avg', 'max'])
    def forward(self, x):
        x_out = x.view((-1, self.num_segments, x.size(1), x.size(2) * x.size(3)))
        x_out = self.gate(x_out)
        x_out = x.view(x.size())
        return x 
    
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
    
class TCBAM(nn.Module):
    def __init__(self, num_segments, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(TCBAM, self).__init__()
        self.num_segments = num_segments
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.TemporalGate = TemporalGate(num_segments)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        x_out = self.TemporalGate(x_out)
        return x_out
    
    
class ConcatShiftGate(nn.Module):
    def __init__(self, num_segments, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ConcatShiftGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        
        size = x.size()
        # (N*T, C, H, W)
        x = x.view((-1, self.num_segments) + size[1:])
        # (N, T, C, H, W)
        x_shift = torch.zeros_like(x)
        x_shift[:, 1:, ...] = x[:, :-1, ...]
        x = x.view(size)
        x_shift = x_shift.view(size)
        
        
        return x * scale + x_shift * (1.0-scale)
    
    
    
class _NonLocalBlockND(nn.Module):
    def __init__(self, num_segments, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        self.num_segments = num_segments

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        # input (N* T, C, H, W)
        original_size = x.size()

        batch_size = x.size(0) // self.num_segments

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        # g_x (N, C//2, T * H * W)
        g_x = g_x.permute(0, 2, 1)
        # (N, T * H * W, C//2)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # (N, T * H* W, C//2)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # (N, C//2, T * H * W)
        f = torch.matmul(theta_x, phi_x)
        # (N, T*H*W, T*H*W)
        N = f.size(-1)
        
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        # (N, T*H*W, C//2)
        y = y.permute(0, 2, 1).contiguous()
        # (N, C//2, T*H*W)
        y = y.view(original_size[0] , -1, original_size[2], original_size[3])
        # (N * T, C//2, H, W)
        W_y = self.W(y)
        # (N * T, C, H, W)
        z = W_y + x

        return z.view(original_size)


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, num_segments, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(num_segments, in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, num_segments, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(num_segments, in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(num_segments, in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
        
        