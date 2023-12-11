import torch
from torch import nn
from easyvolcap.engine import EMBEDDERS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import make_buffer

from easyvolcap.models.networks.embedders.positional_encoding_embedder import PositionalEncodingEmbedder


@EMBEDDERS.register_module()
class AnnealPositionalEncodingEmbedder(PositionalEncodingEmbedder):
    def __init__(self,
                 multires,
                 n_steps=80000,
                 periodic_fns=[torch.sin, torch.cos],
                 retain_input=True,

                 # Should only be used for api consistency
                 in_dim: int = 3,  # only for configurting output shape
                 ):
        super().__init__(multires=multires, 
                         periodic_fns=periodic_fns,
                         retain_input=retain_input,
                         in_dim=in_dim)
        self.n_steps = n_steps

    def forward(self, input: torch.Tensor, batch: dotdict = None):
        """ Eases the original positional encoding in each frequency one by one with a cosine.
        Args:
            input (torch.Tensor): (B, N, 3), original sampled 3d xyz points.
            batch (dotdict): stores the raw data and previous outputs, where we can find batch.frac.
        Returns:
            feat (torch.Tensor): (B, N, embed_dim), anneal positional encoded xyz feature.
        """
        sh = input.shape        # B, N, 3
        feat = super().forward(input=input, batch=batch)    # (B, N, embed)
        # compute the annealing weight for each frequency band
        alpha = self.multires * torch.minimum(batch.iter / self.n_steps, torch.tensor(1.0))
        weight = self.cosine_easing_weight(alpha=alpha)     # (multires,)
        weight = weight[None, None, ...].repeat_interleave(len(self.periodic_fns) * sh[-1], dim=-1)     # (1, 1, embed_dim - 3?)
        if self.retain_input:
            return torch.cat([input, weight * feat[..., sh[-1]:]], dim=-1)
        else:
            return weight * feat

    def cosine_easing_weight(self, alpha):
        """ Eases in each frequency one by one with a cosine. This is equivalent to taking
            a Tukey window and sliding it to the right along the frequency spectrum.
        Args:
            alpha (float): will ease in each frequency as alpha goes from 0.0 to num_freqs.
        Returns:
            A 1-d torch.Tensor with num_sample elements containing the window.
        """
        bands = torch.linspace(0., self.multires - 1, self.multires).to(alpha.device)   # (multires,), multires == (embed_dim - 3) // 2?
        x = torch.clip(alpha - bands, 0.0, 1.0)                         # (multires,)
        return 0.5 * (1 + torch.cos(torch.pi * x + torch.pi))           # (multires,)
