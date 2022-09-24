import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d
import einops
from modules.util import UpBlock2d, DownBlock2d


def make_coordinate_grid(spatial_size, type):
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)

    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

    return meshed


def kp2gaussian_3d(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    # mean = kp['value']
    mean = kp

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class rgb_predictor(nn.Module):
    def __init__(self, in_channels, simpled_channel=128, floor_num=8):
        super(rgb_predictor, self).__init__()
        self.floor_num = floor_num
        self.down_conv = nn.Conv2d(in_channels=in_channels, out_channels=simpled_channel, kernel_size=3, padding=1)

    def forward(self, feature):
        """
        Args:
            feature: warp feature: bs * c * h * w
        Returns:
            rgb: bs * h * w * floor_num * e
        """
        feature = self.down_conv(feature)
        feature = einops.rearrange(feature, 'b (c f) h w  -> b c f h w', f=self.floor_num)
        feature = einops.rearrange(feature, 'b c f h w -> b h w f c')
        return feature


class sigma_predictor(nn.Module):
    def __init__(self, in_channels, simpled_channel=128, floor_num=8):
        super(sigma_predictor, self).__init__()
        self.floor_num = floor_num
        self.down_conv = nn.Conv2d(in_channels=in_channels, out_channels=simpled_channel, kernel_size=3, padding=1)

        self.res_conv3d = nn.Sequential(
            ResBlock3d(16, 3, 1),
            nn.BatchNorm3d(16),
            ResBlock3d(16, 3, 1),
            nn.BatchNorm3d(16),
            ResBlock3d(16, 3, 1),
            nn.BatchNorm3d(16)
        )

    def forward(self, feature):
        """
        Args:
            feature: bs * h * w * floor * c, the output of rgb predictor
        Returns:
            sigma: bs * h * w * floor * encode
            point: bs * 5023 * 3
        """
        heatmap = self.down_conv(feature)
        heatmap = einops.rearrange(heatmap, "b (c f) h w -> b c f h w", f=self.floor_num)
        heatmap = self.res_conv3d(heatmap)
        sigma = einops.rearrange(heatmap, "b c f h w -> b h w f c")

        point_dict = {'sigma_map': heatmap}
        # point_pred = einops.rearrange(point_pred, 'b p n -> b n p')
        return sigma, point_dict


class MultiHeadNeRFModel(torch.nn.Module):

    def __init__(self, hidden_size=128, num_encoding_rgb=16, num_encoding_sigma=16):
        super(MultiHeadNeRFModel, self).__init__()
        # self.xyz_encoding_dims = 1 + 1 * 2 * num_encoding_functions + num_encoding_rgb
        self.xyz_encoding_dims = num_encoding_sigma
        self.viewdir_encoding_dims = num_encoding_rgb

        # Input layer (default: 16 -> 128)
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims, hidden_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 3_1 (default: 128 -> 1): Predicts radiance ("sigma")
        self.layer3_1 = torch.nn.Linear(hidden_size, 1)
        # Layer 3_2 (default: 128 -> 32): Predicts a feature vector (used for color)
        self.layer3_2 = torch.nn.Linear(hidden_size, hidden_size // 4)
        self.layer3_3 = torch.nn.Linear(self.viewdir_encoding_dims, hidden_size)

        # Layer 4 (default: 32 + 128 -> 128)
        self.layer4 = torch.nn.Linear(
            hidden_size // 4 + hidden_size, hidden_size
        )
        # Layer 5 (default: 128 -> 128)
        self.layer5 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 6 (default: 128 -> 256): Predicts RGB color
        self.layer6 = torch.nn.Linear(hidden_size, 256)

        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, rgb_in, sigma_in):
        """
        Args:
            x: rgb pred result of Perdict3D
            view: result of LightPredict
        Returns:
        """
        bs, h, w, floor_num, _ = rgb_in.size()
        # x = torch.cat((x, point3D), dim=-1)
        out = self.relu(self.layer1(sigma_in))
        out = self.relu(self.layer2(out))
        sigma = self.layer3_1(out)
        feat_sigma = self.relu(self.layer3_2(out))
        feat_rgb = self.relu(self.layer3_3(rgb_in))
        x = torch.cat((feat_sigma, feat_rgb), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return x, sigma


def volume_render(rgb_pred, sigma_pred):
    """
    Args:
        rgb_pred: result of Nerf, [bs, h, w, floor, rgb_channel]
        sigma_pred: result of Nerf, [bs, h, w, floor, sigma_channel]
    Returns:

    """
    _, _, _, floor, _ = sigma_pred.size()
    c = 0
    T = 0
    for i in range(floor):
        sigma_mid = torch.nn.functional.relu(sigma_pred[:, :, :, i, :])
        T = T + (-sigma_mid)
        c = c + torch.exp(T) * (1 - torch.exp(-sigma_mid)) * rgb_pred[:, :, :, i, :]
    c = einops.rearrange(c, 'b h w c -> b c h w')
    return c


class RenderModel(nn.Module):
    def __init__(self, in_channels, simpled_channel_rgb, simpled_channel_sigma, floor_num, hidden_size):
        super(RenderModel, self).__init__()
        self.rgb_predict = rgb_predictor(in_channels=in_channels, simpled_channel=simpled_channel_rgb,
                                         floor_num=floor_num)
        self.sigma_predict = sigma_predictor(in_channels=in_channels, simpled_channel=simpled_channel_sigma,
                                             floor_num=floor_num)
        num_encoding_rgb, num_encoding_sigma = simpled_channel_rgb // floor_num, simpled_channel_sigma // floor_num
        self.nerf_module = MultiHeadNeRFModel(hidden_size=hidden_size, num_encoding_rgb=num_encoding_rgb,
                                              num_encoding_sigma=num_encoding_sigma)
        self.mini_decoder = nn.Sequential(
            UpBlock2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            UpBlock2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, feature):
        rgb_in = self.rgb_predict(feature)
        # sigma_in, point_dict = self.sigma_predict(feature.detach())
        sigma_in, point_dict = self.sigma_predict(feature)
        rgb_out, sigma_out = self.nerf_module(rgb_in, sigma_in)
        render_result = volume_render(rgb_out, sigma_out)
        render_result = torch.sigmoid(render_result)
        mini_pred = self.mini_decoder(render_result)
        out_dict = {'render': render_result, 'mini_pred': mini_pred, 'point_pred': point_dict}
        return out_dict
