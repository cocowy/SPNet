from core.update import BasicUpdateBlock
from core.extractor import BasicEncoder
from core.corr import CorrBlock
from core.utils.utils import bilinear_sampler, coords_grid, upflow8
from core.submodule import *

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.neighbor = 3
        self.args.corr_levels = levels = 2
        self.args.corr_radius = 4

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        self.gen_kernl = nn.Sequential(
            nn.Conv2d(hdim + cdim, 64, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, levels * self.neighbor * self.neighbor, 1, 1, 0),  # 4: num_levels, 9: kernel_size ** 2
        )


        ## 0609 modified by Longguang ##
        self.register_parameter('x', torch.Tensor([-1, 0, 1]).view(1, 1, 3, 1, 1).repeat([1, 1, 3, 1, 1]))
        self.register_parameter('y', torch.Tensor([-1, 0, 1]).repeat_interleave(3).view(1, 1, -1, 1, 1))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_coords(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, 8 * H, 8 * W)

    def GetDisparityPlane(self, ab, neighbor=3):
        b, _, h, w = ab.shape

        ab_pad = torch.nn.functional.pad(ab, [1, 1, 1, 1], mode='replicate')
        ab_neighbor = F.unfold(ab_pad, [neighbor, neighbor]).view(b, 2, -1, h, w)  # b, 2, n*n, h, w
        a, b = torch.split(ab_neighbor, [1, 1], dim=1)  # b, 1, n*n, h, w

        ## 0609 modified by Longguang ##
        disp_neighbor = a * self.x + b * self.y  # b, 1, n*n, h, w

        return disp_neighbor.squeeze(1)  # disp: [b, n*n, h, w]

    def forward(self, image, iters=12,  training=True):
        """ Estimate optical flow between pair of frames """
        image1 = image[0]
        image2 = image[1]

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])
        cnet = self.cnet(image1)


        # build lookup table
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        # run the context network
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        # initialization
        coords0, coords1 = self.initialize_coords(image1)  # b, 1, h, w
        ab = torch.zeros_like(coords0).repeat(1, 2, 1, 1)  # b, 3, h, w

        # initial disparity
        b, _, h, w = fmap1.shape

        corr = corr_fn.corr_pyramid[0].view(b, h, w, -1)
        corr = torch.tril(corr) - torch.tril(corr, -192//8)
        corr[corr == 0] = -1e5
        att = corr.softmax(-1)

        index = (torch.arange(w)).view(1, 1, 1, -1).to(corr.device).float()
        disp_ini = (att * (index.transpose(-1, -2) - index)).sum(-1, keepdim=True).view(b, 1, h, w)
        coords1 = coords1 - disp_ini

        # iterations
        disp_predictions = []
        disp_predictions.append(F.interpolate(disp_ini, scale_factor=8, mode='bilinear').squeeze(1) * 8)

        for itr in range(iters):
            disp = coords0 - coords1

            # generate slanted planes
            disp_neighbor = self.GetDisparityPlane(ab)              # b, n*n, h, w

            # lookup
            corr = corr_fn(coords1.detach() - disp_neighbor)
            b, l, r, n, h, w = corr.shape                           # b, levels, 2r+1, n*n, h, w
            corr = corr.view(b, -1, 3, 3, h, w).permute(0, 1, 4, 2, 5, 3)
            corr = corr.contiguous().view(b, -1, h*3, w*3)
            corr_new = F.unfold(corr, kernel_size=3, stride=3, padding=6, dilation=4)
            corr_new = corr_new.view(b, -1, h+2, w+2)[..., 1:-1, 1:-1]

            ## shift 'corr'
            # corr_new = []
            # for i in range(int(math.sqrt(n))):
            #     for j in range(int(math.sqrt(n))):
            #         idx = i * int(math.sqrt(n)) + j
            #         if i != 1 or j != 1:
            #             pad_top = max(i - 1, 0)
            #             pad_down = max(1 - i, 0)
            #             pad_left = max(j - 1, 0)
            #             pad_right = max(1 - j, 0)
            #             start_x = max(1 - i, 0)
            #             start_y = max(1 - j, 0)

            #             corr_slice = corr[:, idx:idx + 1, :, :]
            #             corr_slice = F.pad(corr_slice, [pad_left, pad_right, pad_top, pad_down], mode='replicate')
            #             corr_slice = corr_slice[:, :, start_x:start_x + h, start_y:start_y + w]
            #             corr_new.append(corr_slice)
            #         else:
            #             corr_slice = corr[:, idx:idx + 1, :, :]
            #             corr_new.append(corr_slice)

            # corr_new = torch.cat(corr_new, 1).view(b, l, r, n, h, w)
            corr_new = corr_new.view(b, l, r, n, h, w)
            ## dynamic conv
            kernel = self.gen_kernl(torch.cat([net, inp], dim=1)).view(b, l, 1, n, h, w).softmax(3)
            corr = (corr_new * kernel).sum(3).view(b, -1, h, w)

            # GRU for regression
            net, up_mask, delta_ab, delta_disp, occ_mask = self.update_block(net, inp, corr, disp)
            coords1 = coords0 - (disp * occ_mask + delta_disp * (1 - occ_mask))
            ab = ab + delta_ab

            # upsample predictions
            disp_up = self.upsample_flow(coords0 - coords1, up_mask)[:, 0, ...]

            disp_predictions.append(disp_up)

        if not training:
            return disp, disp_up
        if training:
            return disp_predictions