import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 3, 3, padding=1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)
        self.convc1 = nn.Sequential(
            nn.Conv2d(cor_planes, 256, 1, padding=0),
            nn.BatchNorm2d(256)
        )
        self.convc2 = nn.Sequential(
            nn.Conv2d(256, 192, 3, padding=1),
            nn.BatchNorm2d(192)
        )
        self.convf1 = nn.Sequential(
            nn.Conv2d(1, 128, 7, padding=3),
            nn.BatchNorm2d(128)
        )
        self.convf2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv = nn.Conv2d(64+192, 128-1, 3, padding=1)
        self.relu = torch.nn.LeakyReLU(0.1, True)

    def forward(self, corr, disp):
        cor = self.relu(self.convc1(corr))
        cor = self.relu(self.convc2(cor))
        flo = self.relu(self.convf1(disp))
        flo = self.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.relu(self.conv(cor_flo))
        return torch.cat([out, disp], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        #
        self.occ_mask = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.ab_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 2, 3, padding=1),
        )

        self.disp_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 24, 3, padding=1, bias=False),
        )

        self.register_buffer('index', torch.arange(24).view(1, -1, 1, 1).float())

    def forward(self, net, inp, corr, disp, upsample=True):
        motion_features = self.encoder(corr, disp)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)

        # ab
        ab = self.ab_head(net)

        # occ_mask
        occ_mask = self.occ_mask(net)

        # residual disp
        delta_disp = (self.disp_head(net).softmax(1) * self.index).sum(1, keepdim=True)

        # scale mask to balance gradients
        mask = .25 * self.mask(net)

        return net, mask, ab, delta_disp, occ_mask
