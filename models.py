import torch
from torch import nn
from torchvision.models import densenet201


class MixSepConvBlock(nn.Module):
    """
    https://arxiv.org/abs/1907.09595
    """

    def __init__(self, in_features, out_features, stride=1, dilation=1, res=False, bias=True,
                 batch_norm_activation=True, activation=None):
        super().__init__()
        self.in_features = in_features
        self.dw3 = nn.Conv2d(in_features // 4, in_features // 4, groups=in_features // 4, kernel_size=3,
                             padding=1, stride=stride, dilation=dilation, bias=bias)
        self.dw5 = nn.Conv2d(in_features // 2, in_features // 2, groups=in_features // 2, kernel_size=5,
                             padding=2, stride=stride, dilation=dilation, bias=bias)
        self.dw7 = nn.Conv2d(in_features // 4, in_features // 4, groups=in_features // 4, kernel_size=7,
                             padding=3, stride=stride, dilation=dilation, bias=bias)

        seq = [nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)]
        if batch_norm_activation:
            seq.extend([
                nn.BatchNorm2d(out_features),
                activation or nn.ReLU(inplace=True),
            ])
        self.conv = nn.Sequential(*seq)
        self.res = res

    def forward(self, x):
        q = self.in_features // 4
        x3 = self.dw3(x[:, :q, :, :])
        x5 = self.dw5(x[:, q:3 * q, :, :])
        x7 = self.dw7(x[:, 3 * q:, :, :])
        y = self.conv(torch.cat([x3, x5, x7], dim=1))
        if self.res:
            return x + y
        return y


class CoreFPN(nn.Module):

    def __init__(self, num_filters=256, pretrained=True, **kwargs):
        super().__init__()
        densenet = densenet201(pretrained=pretrained, **kwargs).features

        self.features_enc0 = nn.Sequential(densenet.conv0,
                                           densenet.norm0,
                                           densenet.relu0)
        self.features_pool0 = densenet.pool0
        self.features_enc1 = densenet.denseblock1
        self.features_enc2 = densenet.denseblock2
        self.features_enc3 = densenet.denseblock3

        self.features_tr1 = densenet.transition1
        self.features_tr2 = densenet.transition2

        self.lateral3 = nn.Sequential(nn.Conv2d(1792, num_filters, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(num_filters),
                                      nn.ReLU(inplace=True))
        self.lateral2 = nn.Sequential(nn.Conv2d(512, num_filters, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(num_filters),
                                      nn.ReLU(inplace=True))
        self.lateral0 = nn.Sequential(nn.Conv2d(64, num_filters // 4, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(num_filters // 4),
                                      nn.ReLU(inplace=True))

        self.upsample_map3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample_map2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.td1 = MixSepConvBlock(num_filters, num_filters, res=True)
        self.td2 = MixSepConvBlock(num_filters, num_filters, res=True)

    def forward(self, x):
        enc0 = self.features_enc0(x)

        pooled = self.features_pool0(enc0)

        enc1 = self.features_enc1(pooled)
        tr1 = self.features_tr1(enc1)

        enc2 = self.features_enc2(tr1)
        tr2 = self.features_tr2(enc2)

        enc3 = self.features_enc3(tr2)

        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)

        map3 = lateral3
        map2 = self.td1(lateral2 + self.upsample_map3(map3))
        map1 = self.td2(enc1 + self.upsample_map2(map2))
        map0 = self.lateral0(enc0)

        return map0, map1, map2, map3  # scales /2, /4, /8, /16


class TigerFPN(nn.Module):
    def __init__(self, num_filters=256, num_out=15):
        super().__init__()
        self.fpn = CoreFPN(num_filters=num_filters)
        self.upsample_map3 = nn.UpsamplingNearest2d(scale_factor=8)
        self.upsample_map2 = nn.UpsamplingNearest2d(scale_factor=4)
        self.upsample_map1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample_final = nn.UpsamplingNearest2d(scale_factor=2)
        self.stacked_conv = nn.Sequential(
            MixSepConvBlock(num_filters * 3, num_filters, res=False, bias=False),
            MixSepConvBlock(num_filters, num_filters, res=True, bias=False),
            MixSepConvBlock(num_filters, num_filters, res=True, bias=False)
        )
        self.conv_m0 = MixSepConvBlock(num_filters // 4, num_filters, res=False, bias=True)
        self.final_conv = nn.Conv2d(num_filters, num_out, kernel_size=1)
        self.linear = nn.Linear(num_filters, out_features=num_out, bias=True)

    def forward(self, x):
        m0, m1, m2, m3 = self.fpn(x)

        m3 = self.upsample_map3(m3)
        m2 = self.upsample_map2(m2)
        m1 = self.upsample_map1(m1)
        stacked = torch.cat(tensors=(m1, m2, m3),
                            dim=1)
        stacked = self.stacked_conv(stacked)
        lin_features = torch.mean(stacked, dim=(2, 3))
        logits = self.linear(lin_features)
        stacked = self.upsample_final(stacked + self.conv_m0(m0))
        final = self.final_conv(stacked)
        return final, logits


if __name__ == '__main__':
    t = torch.rand(1, 3, 256, 256).cuda()
    m = TigerFPN().cuda()
    x, l = m(t)
    print(x.size(), l.size())
