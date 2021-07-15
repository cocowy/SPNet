import torch
import torch.nn.functional as F
from core.utils_lab.utils import bilinear_sampler, coords_grid
import math
try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, w2 = corr.shape
        corr = corr.view(-1, 1, w2)

        self.corr_pyramid.append(corr.view(batch*h1*w1, 1, 1, w2))                  #bhw1, 1, 1, w2
        for i in range(self.num_levels - 1):
            corr = F.avg_pool1d(corr, 2, stride=2)                                  # w2-->w2/2,w2/4,w2/4
            self.corr_pyramid.append(corr.view(batch*h1*w1, 1, 1, -1))

        ## 0609 modified by Longguang ##
        r = self.radius
        self.dx = torch.linspace(-r, r, 2*r+1).view(1, 2*r+1, 1, 1).to(corr.device)

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            centroid_lvl = coords.reshape(batch*h1*w1, 1, -1, 1) / 2**i                     # bhw, 1, n*n, 1
            
            ## 0609 modified by Longguang ##
            coords_lvl = centroid_lvl + self.dx                                             # bhw, 2r+1, n*n, 1
            coords_lvl = torch.cat([coords_lvl, torch.zeros_like(coords_lvl)], -1)          # bhw, 2r+1, n*n, 2

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape

        fmap1 = fmap1.permute(0, 2, 3, 1).contiguous()                              # b * h * w * c
        fmap2 = fmap2.permute(0, 2, 1, 3).contiguous()                              # b * h * c * w

        corr = torch.matmul(fmap1, fmap2)                                           # b * h * w * w

        return corr / torch.sqrt(torch.tensor(dim).float())

