import torch

from torch import nn
from torch.nn import functional as F


# !: IMPORT
from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.net_utils import get_function


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_actvn=nn.BatchNorm2d):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_actvn(out_channels)
        self.relu = nn.ReLU(inplace=True)  # might pose problem for pass through optimization, albeit faster

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_actvn=nn.BatchNorm3d):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_actvn(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


@REGRESSORS.register_module()
class FeatureNet(nn.Module):
    def __init__(self, norm_actvn=nn.BatchNorm2d, test_using_train: bool = True):
        super(FeatureNet, self).__init__()
        norm_actvn = getattr(nn, norm_actvn) if isinstance(norm_actvn, str) else norm_actvn

        self.conv0 = nn.Sequential(
            ConvBnReLU(3, 8, 3, 1, 1, norm_actvn=norm_actvn),
            ConvBnReLU(8, 8, 3, 1, 1, norm_actvn=norm_actvn))
        self.conv1 = nn.Sequential(
            ConvBnReLU(8, 16, 5, 2, 2, norm_actvn=norm_actvn),
            ConvBnReLU(16, 16, 3, 1, 1, norm_actvn=norm_actvn))
        self.conv2 = nn.Sequential(
            ConvBnReLU(16, 32, 5, 2, 2, norm_actvn=norm_actvn),
            ConvBnReLU(32, 32, 3, 1, 1, norm_actvn=norm_actvn))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

        self.out_dims = [32, 16, 8]  # output dimensionality
        self.scales = [0.25, 0.5, 1.0]
        self.size_pad = 4  # input size should be divisible by 4
        self.test_using_train = test_using_train

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + y

    def forward(self, x: torch.Tensor):
        # x: (B, S, C, H, W) or (B, C, H, W) or (C, H, W)
        # Remember input shapes
        sh = x.shape
        x = x.view(-1, *sh[-3:])  # (B, C, H, W)

        # NOTE: We assume normalized -1, 1 rgb input for feature_net

        # Actual conv net
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        feat2 = self.toplayer(conv2)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))
        feat0 = self._upsample_add(feat1, self.lat0(conv0))
        feat1 = self.smooth1(feat1)
        feat0 = self.smooth0(feat0)

        # Restore original shapes
        feat2 = feat2.view(sh[:-3] + feat2.shape[-3:])
        feat1 = feat1.view(sh[:-3] + feat1.shape[-3:])
        feat0 = feat0.view(sh[:-3] + feat0.shape[-3:])
        return feat2, feat1, feat0  # level0, level1, level2

    def train(self, mode: bool):
        if not mode and self.test_using_train: return
        super().train(mode)


@REGRESSORS.register_module()
class CostRegNet(nn.Module):
    # TODO: compare the results of nn.BatchNorm3d and nn.InstanceNorm3d
    def __init__(self, in_channels, norm_actvn=nn.BatchNorm3d, dpt_actvn=nn.Identity, use_vox_feat=True):
        super(CostRegNet, self).__init__()
        norm_actvn = getattr(nn, norm_actvn) if isinstance(norm_actvn, str) else norm_actvn
        self.dpt_actvn = get_function(dpt_actvn)

        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_actvn=norm_actvn)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_actvn=norm_actvn)
        self.conv2 = ConvBnReLU3D(16, 16, norm_actvn=norm_actvn)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_actvn=norm_actvn)
        self.conv4 = ConvBnReLU3D(32, 32, norm_actvn=norm_actvn)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_actvn=norm_actvn)
        self.conv6 = ConvBnReLU3D(64, 64, norm_actvn=norm_actvn)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_actvn(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_actvn(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_actvn(8))
        self.depth_conv = nn.Sequential(nn.Conv3d(8, 1, 3, padding=1, bias=False))

        self.use_vox_feat = use_vox_feat
        if use_vox_feat:
            self.feat_conv = nn.Sequential(nn.Conv3d(8, 8, 3, padding=1, bias=False))

        self.size_pad = 8  # input size should be divisible by 4
        self.out_dim = 8

    def forward(self, x: torch.Tensor):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        depth = self.depth_conv(x)
        depth = self.dpt_actvn(depth.squeeze(1))  # softplus might change dtype

        if self.use_vox_feat:
            feat = self.feat_conv(x)
            return feat, depth
        else:
            return depth


@REGRESSORS.register_module()
class MinCostRegNet(nn.Module):
    def __init__(self, in_channels, norm_actvn=nn.BatchNorm3d, dpt_actvn=nn.Identity):
        super(MinCostRegNet, self).__init__()
        norm_actvn = getattr(nn, norm_actvn) if isinstance(norm_actvn, str) else norm_actvn
        self.dpt_actvn = get_function(dpt_actvn)

        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_actvn=norm_actvn)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_actvn=norm_actvn)
        self.conv2 = ConvBnReLU3D(16, 16, norm_actvn=norm_actvn)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_actvn=norm_actvn)
        self.conv4 = ConvBnReLU3D(32, 32, norm_actvn=norm_actvn)

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_actvn(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_actvn(8))

        self.depth_conv = nn.Sequential(nn.Conv3d(8, 1, 3, padding=1, bias=False))
        self.feat_conv = nn.Sequential(nn.Conv3d(8, 8, 3, padding=1, bias=False))
        self.out_dim = 8
        self.size_pad = 4  # input should be divisible by 4

    def forward(self, x, use_vox_feat=True):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        depth = self.depth_conv(x)
        depth = self.dpt_actvn(depth.squeeze(1))

        if not use_vox_feat: feat = None
        else: feat = self.feat_conv(x)
        return feat, depth


# ? This could be refactored

class NeRF(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16 + 3):
        """
        """
        super(NeRF, self).__init__()
        self.hid_n = hid_n
        self.agg = FeatureAgg(feat_ch)
        self.lr0 = nn.Sequential(nn.Linear(16, hid_n),
                                 nn.ReLU())
        self.lrs = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_n, hid_n), nn.ReLU()) for i in range(0)
        ])
        self.sigma = nn.Sequential(nn.Linear(hid_n, 1), nn.Softplus())
        self.color = nn.Sequential(
            nn.Linear(64 + 16 + feat_ch + 4, hid_n),
            nn.ReLU(),
            nn.Linear(hid_n, 1),
            nn.ReLU())
        self.lr0.apply(weights_init)
        self.lrs.apply(weights_init)
        self.sigma.apply(weights_init)
        self.color.apply(weights_init)

    def forward(self, vox_feat, img_feat_rgb_dir):
        B, N_points, N_views = img_feat_rgb_dir.shape[:-1]
        img_feat = self.agg(img_feat_rgb_dir)
        S = img_feat_rgb_dir.shape[2]

        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1)

        x = self.lr0(vox_img_feat)
        for i in range(len(self.lrs)):
            x = self.lrs[i](x)
        sigma = self.sigma(x)
        x = torch.cat((x, vox_img_feat), dim=-1)
        x = x.view(B, -1, 1, x.shape[-1]).repeat(1, 1, S, 1)
        x = torch.cat((x, img_feat_rgb_dir), dim=-1)
        color_weight = F.softmax(self.color(x), dim=-2)
        color = torch.sum((img_feat_rgb_dir[..., -7:-4] * color_weight), dim=-2)
        return torch.cat([color, sigma], dim=-1)


@REGRESSORS.register_module()
class FeatureAgg(nn.Module):
    def __init__(self, feat_ch, viewdir_agg=True):
        """
        """
        super(FeatureAgg, self).__init__()
        self.feat_ch = feat_ch

        # Layered ENeRF ignores viewdir during vanilla xyz embedding
        self.viewdir_agg = viewdir_agg
        if self.viewdir_agg:
            self.view_fc = nn.Sequential(
                nn.Linear(4, feat_ch),
                nn.ReLU(),
            )
            self.view_fc.apply(weights_init)
        self.global_fc = nn.Sequential(
            nn.Linear(feat_ch * 3, 32),
            nn.ReLU(),
        )

        self.agg_w_fc = nn.Sequential(
            nn.Linear(32, 1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.global_fc.apply(weights_init)
        self.agg_w_fc.apply(weights_init)
        self.fc.apply(weights_init)

        self.out_dim = 16

    def forward(self, img_feat_rgb_dir: torch.Tensor):
        # Prepare shapes
        img_feat_rgb_dir = img_feat_rgb_dir.permute(0, 2, 1, 3)  # B, S, P, C -> B, P, S, C

        B, S = len(img_feat_rgb_dir), img_feat_rgb_dir.shape[-2]

        if self.viewdir_agg:
            view_feat = self.view_fc(img_feat_rgb_dir[..., self.feat_ch:])
            img_feat_rgb = img_feat_rgb_dir[..., :self.feat_ch] + view_feat
        else:
            img_feat_rgb = img_feat_rgb_dir[..., :self.feat_ch]

        var_feat = torch.var(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).expand(-1, -1, S, -1)
        avg_feat = torch.mean(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).expand(-1, -1, S, -1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1)
        global_feat = self.global_fc(feat)
        agg_w = F.softmax(self.agg_w_fc(global_feat), dim=-2)
        im_feat = (global_feat * agg_w).sum(dim=-2)
        return self.fc(im_feat)  # B, P, C


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, padding_mode='reflect')


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode='reflect')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2,
                              padding_mode='reflect')
        self.bn = nn.InstanceNorm2d(num_out_layers, track_running_stats=False, affine=True)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


@REGRESSORS.register_module()
class ResUNet(nn.Module):
    def __init__(self,
                 encoder='resnet34',
                 coarse_out_ch=32,
                 fine_out_ch=32,
                 norm_actvn=nn.InstanceNorm2d,
                 coarse_only=False,
                 **kwargs
                 ):

        super(ResUNet, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        self.coarse_only = coarse_only
        if self.coarse_only:
            fine_out_ch = 0
        self.coarse_out_ch = coarse_out_ch
        self.fine_out_ch = fine_out_ch
        out_ch = coarse_out_ch + fine_out_ch

        # original
        layers = [3, 4, 6, 3]
        self._norm_layer = norm_actvn
        self.dilation = 1
        block = BasicBlock
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False, padding_mode='reflect')
        self.bn1 = norm_actvn(self.inplanes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # decoder
        self.upconv3 = upconv(filters[2], 128, 3, 2)
        self.iconv3 = conv(filters[1] + 128, 128, 3, 1)
        self.upconv2 = upconv(128, 64, 3, 2)
        self.iconv2 = conv(filters[0] + 64, out_ch, 3, 1)

        # fine-level conv
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1, 1)

        self.out_dims = [32, 32]  # output dimensionality
        self.scales = [0.125, 0.25]
        self.size_pad = 1  # input size should be divisible by 4

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, track_running_stats=False, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_out = self.out_conv(x)

        if self.coarse_only:
            x_coarse = x_out
            x_fine = None
            return [x_coarse]
        else:
            x_fine = x_out[:, :self.fine_out_ch, :]
            x_coarse = x_out[:, -self.coarse_out_ch:, :]
            return [x_fine, x_coarse]
