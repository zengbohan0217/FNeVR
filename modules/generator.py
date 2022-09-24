import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork
from modules.nerf_verts_util import RenderModel


class SPADE_layer(nn.Module):
    def __init__(self, norm_channel, label_channel):
        super(SPADE_layer, self).__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_channel, affine=False)
        hidden_channel = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_channel, hidden_channel, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(hidden_channel, norm_channel, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_channel, norm_channel, kernel_size=3, padding=1)

    def forward(self, x, modulation_in):
        normalized = self.param_free_norm(x)
        modulation_in = F.interpolate(modulation_in, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(modulation_in)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class SPADE_block(nn.Module):
    def __init__(self, norm_channel, label_channel, out_channel):
        super(SPADE_block, self).__init__()
        self.SPADE_0 = SPADE_layer(norm_channel, label_channel)
        self.relu_0 = nn.ReLU()
        self.conv_0 = nn.Conv2d(norm_channel, norm_channel, kernel_size=3, padding=1)
        self.SPADE_1 = SPADE_layer(norm_channel, label_channel)
        self.relu_1 = nn.ReLU()
        self.conv_1 = nn.Conv2d(norm_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x, modulation_in):
        out = self.SPADE_0(x, modulation_in)
        out = self.relu_0(out)
        out = self.conv_0(out)
        out = self.SPADE_1(out, modulation_in)
        out = self.relu_1(out)
        out = self.conv_1(out)
        return out


class SPADE_decoder(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(SPADE_decoder, self).__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.seg_conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.SPADE_0 = SPADE_block(in_channel, mid_channel, in_channel // 4)
        self.up_0 = nn.UpsamplingBilinear2d(scale_factor=2)
        in_channel = in_channel // 4
        self.SPADE_1 = SPADE_block(in_channel, mid_channel, in_channel // 4)
        self.up_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        in_channel = in_channel // 4
        self.SPADE_2 = SPADE_block(in_channel, mid_channel, in_channel)
        self.SPADE_3 = SPADE_block(in_channel, mid_channel, in_channel)
        self.final = nn.Sequential(
            nn.Conv2d(in_channel, 3, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        seg = self.seg_conv(x)
        x = self.SPADE_0(x, seg)
        x = self.up_0(x)
        x = self.SPADE_1(x, seg)
        x = self.up_1(x)
        x = self.SPADE_2(x, seg)
        x = self.SPADE_3(x, seg)
        x = self.final(x)
        return x


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(x, modulation_in):
    assert (x.size()[:2] == modulation_in.size()[:2])
    size = x.size()
    style_mean, style_std = calc_mean_std(modulation_in)
    content_mean, content_std = calc_mean_std(x)

    normalized_feat = (x - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class AdaIN_layer(nn.Module):
    def __init__(self, norm_channel, label_channel):
        super(AdaIN_layer, self).__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_channel, affine=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_channel, norm_channel, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, modulation_in):
        normalized = self.param_free_norm(x)
        modulation_in = self.mlp_shared(modulation_in)
        out = adaptive_instance_normalization(normalized, modulation_in)
        return out


class OcclusionAwareGenerator_SPADE(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, render_params=None,
                 estimate_jacobian=False):
        super(OcclusionAwareGenerator_SPADE, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))

        self.Render_model = RenderModel(in_channels=in_features, **render_params)
        self.decoder = SPADE_decoder(in_channel=in_features * 2, mid_channel=128)

        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            output_dict["deformed"] = self.deform_input(source_image, deformation)

        # render part
        render_result = self.Render_model(feature=out)
        output_dict['render'] = render_result['mini_pred']
        output_dict['point_pred'] = render_result['point_pred']
        out = torch.cat((out, render_result['render']), dim=1)
        # out = self.merge_conv(out)

        # Decoding part
        out = self.decoder(out)

        output_dict["prediction"] = out

        return output_dict
