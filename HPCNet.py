"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from Blanced_attention import BlancedAttention, BlancedAttention_CAM_SAM_ADD
from einops import rearrange
from ALformer import RestormerBlock


##########################################################################
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):  # 不改变size的conv
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def st_conv(in_channels, out_channels, kernel_size, bias=False, stride=2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# contrast-aware channel attention module
def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Global context Layer
# class GCLayer(nn.Module):
# def __init__(self, channel, reduction=16, bias=False):
# super(GCLayer, self).__init__()
# # global average pooling: feature --> point
# #self.avg_pool = nn.AdaptiveAvgPool2d(1)
# # feature channel downscale and upscale --> channel weight
# self.conv_phi = nn.Conv2d(channel, 1, 1, stride=1,padding=0, bias=False)
# self.softmax = nn.Softmax(dim=1)

# self.conv_du = nn.Sequential(
# nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
# nn.ReLU(inplace=True),
# nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
# nn.Sigmoid()
# )

# def forward(self, x):
# b, c, h, w = x.size()
# #y = self.avg_pool(x)
# y_1 = self.conv_phi(x).view(b, 1, -1).permute(0, 2, 1).contiguous()### b,hw,1
# y_1_att = self.softmax(y_1)
# print(y_1.size)
# x_1 = x.view(b, c, -1)### b,c,hw
# mul_context = torch.matmul(x_1, y_1_att)#### b,c,1
# mul_context = mul_context.view(b, c, 1, -1)

# y = self.conv_du(mul_context)
# return x * y

##########################################################################
## Semantic-guidance Texture Enhancement Module
class STEM(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias=False):
        super(STEM, self).__init__()
        # global average pooling: feature --> point

        # self.down_img = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)#DownSample1()
        self.conv_stem0 = st_conv(3, n_feat, kernel_size, bias=bias)
        # self.conv_stem1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv_stem2 = conv(3, n_feat, kernel_size, bias=bias)
        self.conv_stem3 = conv(3, n_feat, kernel_size, bias=bias)
        # self.CA_fea = CALayer(n_feat, reduction, bias=bias)

    def forward(self, img_rain, res, img):
        # img_down = self.down_img(img_rain)
        img_down = self.conv_stem0(img_rain)
        # img_down_fea = self.conv_stem1(img_down)
        res_fea = self.conv_stem2(res)
        rain_mask = torch.sigmoid(res_fea)
        # rain_mask = self.CA_fea(res_fea)
        att_fea = img_down * rain_mask + img_down
        img_fea = self.conv_stem3(img)
        return att_fea + img_fea
        # return torch.cat([img_down_fea * rain_mask, img_fea],1)


##########################################################################
## S2FB
class S2FB_2(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_2, self).__init__()
        self.DSC = depthwise_separable_conv(n_feat * 2, n_feat)
        # self.CON_FEA = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        # self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        # self.CA_fea = CCALayer(n_feat, reduction, bias=bias)

    def forward(self, x1, x2):
        FEA_1 = self.DSC(torch.cat((x1, x2), 1))
        # FEA_2 = self.CON_FEA(torch.cat((x1,x2), 1))
        # resin = FEA_1 + FEA_2
        res = self.CA_fea(FEA_1) + x1
        # res += resin
        return res  # x1 + resin


##########################################################################
## S2FB
class S2FB_4(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_4, self).__init__()
        # self.CA_fea1 = CALayer(n_feat, reduction, bias=bias)
        # self.CA_fea2 = CALayer(n_feat, reduction, bias=bias)
        # self.CA_fea3 = CALayer(n_feat, reduction, bias=bias)
        self.DSC = depthwise_separable_conv(n_feat * 4, n_feat)
        # self.CON_FEA = nn.Conv2d(n_feat*3, n_feat, kernel_size=1, bias=bias)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        # self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        # self.CA_fea = CCALayer(n_feat, reduction, bias=bias)

    def forward(self, x1, x2, x3, x4):
        FEA_1 = self.DSC(torch.cat((x1, x2, x3, x4), 1))
        # FEA_2 = self.CON_FEA(torch.cat((x2,x3,x4), 1))
        # resin = FEA_1 + FEA_2
        res = self.CA_fea(FEA_1) + FEA_1
        # res += resin
        # res1= self.CA_fea1(FEA_1)
        # res2= self.CA_fea2(x3)
        # res3= self.CA_fea3(x4)
        return res  # x1 + res


class S2FB_p(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_p, self).__init__()
        # self.CA_fea1 = CALayer(n_feat, reduction, bias=bias)
        # self.CA_fea2 = CALayer(n_feat, reduction, bias=bias)
        # self.CA_fea3 = CALayer(n_feat, reduction, bias=bias)
        self.DSC1 = depthwise_separable_conv(n_feat * 2, n_feat)
        self.DSC2 = depthwise_separable_conv(n_feat * 2, n_feat)
        self.DSC3 = depthwise_separable_conv(n_feat * 2, n_feat)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        # self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        # self.CA_fea = CCALayer(n_feat, reduction, bias=bias)

    def forward(self, x1, x2, x3, x4):
        FEA_34 = self.DSC1(torch.cat((x3, x4), 1))
        FEA_34_2 = self.DSC2(torch.cat((x2, FEA_34), 1))
        FEA_34_2_1 = self.DSC2(torch.cat((x1, FEA_34_2), 1))
        res = self.CA_fea(FEA_34_2_1) + FEA_34_2_1
        # res += resin
        # res1= self.CA_fea1(FEA_1)
        # res2= self.CA_fea2(x3)
        # res3= self.CA_fea3(x4)
        return res  # x1 + res


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


## Enhanced Channel Attention Block (ECAB)
class ECAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(ECAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        # res += x
        return res


## Channel Attention Block (CAB)
class CAB_dsc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB_dsc, self).__init__()
        modules_body = []
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))
        modules_body.append(act)
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        # res += x
        return res


##########################################################################
##---------- Resizing Modules ----------
# class DownSample1(nn.Module):
# def __init__(self):
# super(DownSample, self).__init__()
# self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

# def forward(self, x):
# x = self.down(x)
# return x

'''class DownSample(nn.Module):
    def __init__(self, in_channels):
        # def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class DownSample4(nn.Module):
    def __init__(self, in_channels):
        # def __init__(self, in_channels,s_factor):
        super(DownSample4, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class DownSample8(nn.Module):
    def __init__(self, in_channels):
        # def __init__(self, in_channels,s_factor):
        super(DownSample4, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x'''


# class UpSample1(nn.Module):
# def __init__(self, in_channels):
# #def __init__(self, in_channels,s_factor):
# super(UpSample, self).__init__()
# self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

# def forward(self, x):
# x = self.up(x)
# return x

class UpSample(nn.Module):
    def __init__(self, in_channels):
        # def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class UpSample4(nn.Module):
    def __init__(self, in_channels):
        # def __init__(self, in_channels,s_factor):
        super(UpSample4, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    # def __init__(self, in_channels,s_factor):
    def __init__(self, in_channels):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),  # // : 整除,向下取整
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x * y


##########################################################################
## Long Feature Selection and Fusion Block (LFSFB)
class LFSFB(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias):
        super(LFSFB, self).__init__()
        self.FS = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=False)
        self.act1 = act
        self.FFU = nn.ConvTranspose2d(n_feat, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.act2 = act

    def forward(self, x1, x2):
        res = self.act1(self.FS(x1))
        # res = self.act1(res)
        # print(res.shape)
        res_out = self.act2(self.FFU(x2 + res))
        # res = self.act2(res)
        # print(res.shape)
        return res_out


##########################################################################
## Reconstruction and Reproduction Block (RRB)
class RRB(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias):
        super(RRB, self).__init__()
        self.recon_B = conv(n_feat, 3, kernel_size, bias=bias)
        self.recon_R = conv(n_feat, 3, kernel_size, bias=bias)
        # self.recon_BTOR =  conv(n_feat, 3, kernel_size, bias=bias)
        # self.recon_RTOB = conv(n_feat, 3, kernel_size, bias=bias)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(3, n_feat, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, 3, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        xB = x[0]
        xR = x[1]
        #xBTOR = x[2]
        #xRTOB = x[3]
        recon_B = self.recon_B(xB)  # IB*
        recon_R = self.recon_R(xR)
        #recon_BTOR = self.recon_BTOR(xBTOR)
        #recon_RTOB = self.recon_RTOB(xRTOB)
        #res = self.avg_pool(recon_B + recon_R)
        #res_att = self.conv_du(res)
        #re_rain = recon_B * res_att + recon_R * (1 - res_att)
        re_rain = recon_B + recon_R
        return [recon_B, re_rain, recon_R]


##########################################################################
## Coupled Representation Block (CRB)
class CRB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_rcab):
        super(CRB, self).__init__()

        heads = 2
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'
        num_RB = num_rcab  # number of Restormer Blocks
        self.CAB_r = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.CAB_b = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.down_R = st_conv(n_feat, n_feat, kernel_size, bias=bias)
        self.act1 = act
        modules_body = []
        # modules_body = [CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_rcab)]
        modules_body = [
            RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                           LayerNorm_type=LayerNorm_type) for _ in range(num_RB)]
        self.body = nn.Sequential(*modules_body)

        self.lfsfb = LFSFB(n_feat, kernel_size, act, bias)
        # self.CA_B = SALayer(n_feat, reduction, bias=bias)
        # self.CA_R = SALayer(n_feat, reduction, bias=bias)
        self.CA_B = RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                   LayerNorm_type=LayerNorm_type)
        self.CA_R = RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                   LayerNorm_type=LayerNorm_type)

    def forward(self, x):
        xB = x[0]
        xR = x[1]
        res_down_R = self.act1(self.down_R(xR))
        res_R = self.body(res_down_R)
        xR_res = self.CAB_r(xR) + self.lfsfb(res_down_R, res_R)

        res_BTOR = self.CA_B(xB)
        res_RTOB = self.CA_R(xR_res)
        x[0] = self.CAB_b(xB) - res_BTOR + res_RTOB
        x[1] = xR_res - res_RTOB + res_BTOR
        '''if len(x) == 2:
            x.append(res_BTOR)
            x.append(res_RTOB)'''

        return x


##########################################################################
## Coupled Representation Module (CRM)
class CRM(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_crb, num_rcab):
        super(CRM, self).__init__()
        modules_body = [CRB(n_feat, kernel_size, reduction, bias=bias, act=act, num_rcab=num_rcab) for _ in
                        range(num_crb)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x_B, x_R):
        res = self.body([x_B, x_R])
        # res += x
        return res


##########################################################################
class MODEL(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_crb, num_rcab):
        super(MODEL, self).__init__()

        self.image_fea = conv(3, n_feat, kernel_size, bias=bias)
        #### embedding
        self.shallow_fea_B = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.shallow_fea_R = conv(n_feat, n_feat, kernel_size, bias=bias)

        self.down_B = st_conv(n_feat, n_feat, kernel_size, bias=bias)
        self.down_R = st_conv(n_feat, n_feat, kernel_size, bias=bias)

        self.crm = CRM(n_feat, kernel_size, reduction, act, bias, num_crb, num_rcab)

        # self.lfsfb_B = LFSFB(n_feat, kernel_size, act, bias)
        self.UP_B = nn.Sequential(
            nn.ConvTranspose2d(n_feat, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            act)  # , conv(n_feat, n_feat, 1, bias=bias))
        # self.lfsfb_R = LFSFB(n_feat, kernel_size, act, bias)
        self.UP_R = nn.Sequential(
            nn.ConvTranspose2d(n_feat, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            act)  # , conv(n_feat, n_feat, 1, bias=bias))
        self.UP_BTOR = nn.Sequential(
            nn.ConvTranspose2d(n_feat, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            act)  # , conv(n_feat, n_feat, 1, bias=bias))
        self.UP_RTOB = nn.Sequential(
            nn.ConvTranspose2d(n_feat, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            act)  # , conv(n_feat, n_feat, 1, bias=bias))
        self.rrb = RRB(n_feat, kernel_size, act, bias=bias)

    def forward(self, x):
        x_fea = self.image_fea(x)
        B_fea = self.shallow_fea_B(x_fea)
        R_fea = self.shallow_fea_R(x_fea)

        B_down_fea = self.down_B(B_fea)
        R_down_fea = self.down_R(R_fea)
        # cat_fea = [B_down_fea, R_down_fea]
        [fea_B, fea_R] = self.crm(B_down_fea, R_down_fea)  # wq 改了一下fea_BTOR, fea_RTOB大小写

        # fea_B_fuse = self.lfsfb_B(B_down_fea, fea_B)
        # fea_R_fuse = self.lfsfb_R(R_down_fea, fea_R)
        fea_B_fuse = self.UP_B(fea_B)
        fea_R_fuse = self.UP_R(fea_R)
        # fea_BTOR_fuse = self.UP_BTOR(fea_BTOR)
        # fea_RTOB_fuse = self.UP_RTOB(fea_RTOB)

        [img_B, img_R, streak] = self.rrb([fea_B_fuse, fea_R_fuse])

        # fea_BTOR = fea_BTOR[:, 0:3, :, :]
        # fea_RTOB = fea_RTOB[:, 0:3, :, :]

        return img_B, img_R


##########################################################################
class HPCNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=48, kernel_size=3, reduction=4, num_crb=5, num_rcab=4, bias=False):
        super(HPCNet, self).__init__()

        act = nn.PReLU()
        self.model = MODEL(n_feat, kernel_size, reduction, act, bias, num_crb, num_rcab)

    def forward(self, x_img):  #####b,c,h,w
        # print(x_img.shape)
        imitation, rain_out = self.model(x_img)
        # print(imitation.shape)
        # print(rain_out.shape)
        return [imitation, rain_out]