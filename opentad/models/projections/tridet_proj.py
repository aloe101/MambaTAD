import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import ConvModule, SGPBlock, AffineDropPath, SGPMDBlock
from ..builder import PROJECTIONS
from mamba_ssm.modules.bssdm import Mamba as DBM

from .actionformer_proj import get_sinusoid_encoding


@PROJECTIONS.register_module()
class TriDetProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        sgp_mlp_dim,  # dim in SGP
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        sgp_win_size=[-1] * 6,  # size of local window for mha
        downsample_type="max",  # how to downsample feature in FPN
        k=1.5,
        init_conv_vars=1,  # initialization of gaussian variance for the weight in SGP
        conv_cfg=None,  # kernel_size
        norm_cfg=None,
        path_pdrop=0.0,  # dropout rate for drop path
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        input_noise=0.0,
        mamba_cfg=dict(kernel_size=4, drop_path_rate=0.3, use_mamba_type="dbm"),
        sgpmd = False,
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(sgp_win_size) == (1 + arch[2])

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.scale_factor = 2  # as default

        self.path_pdrop = path_pdrop
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len
        self.sgp_win_size = sgp_win_size
        self.downsample_type = downsample_type
        self.input_noise = input_noise

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)
        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels**0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

        # embedding network using convs
        self.embed = nn.ModuleList()
        for i in range(arch[0]):
            self.embed.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

        # stem network using SGP blocks
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            if sgpmd:
                self.stem.append(
                    SGPMDBlock(
                    out_channels,
                    kernel_size=1,
                    n_ds_stride=1,
                    n_hidden=sgp_mlp_dim,
                    k=k,
                    init_conv_vars=init_conv_vars,
                )
                )
            else:
                self.stem.append(
                    SGPBlock(
                        out_channels,
                        kernel_size=1,
                        n_ds_stride=1,
                        n_hidden=sgp_mlp_dim,
                        k=k,
                        init_conv_vars=init_conv_vars,
                    )
                )
            

        # main branch using SGP blocks with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            if sgpmd:
                self.branch.append(
                    SGPMDBlock(
                    out_channels,
                    kernel_size=self.sgp_win_size[1 + idx],
                    n_ds_stride=self.scale_factor,
                    n_hidden=sgp_mlp_dim,
                    k=k,
                    init_conv_vars=init_conv_vars,
                )
                )
            else:
                self.branch.append(
                SGPBlock(
                    out_channels,
                    kernel_size=self.sgp_win_size[1 + idx],
                    n_ds_stride=self.scale_factor,
                    path_pdrop=self.path_pdrop,
                    n_hidden=sgp_mlp_dim,
                    downsample_type=downsample_type,
                    k=k,
                    init_conv_vars=init_conv_vars,
                )
            )
            # MaskMambaBlock(out_channels, init_conv_vars=init_conv_vars, **mamba_cfg)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)

        # trick, adding noise may slightly increases the variability between input features.
        if self.training and self.input_noise > 0:
            noise = torch.randn_like(x) * self.input_noise
            x += noise

        # embedding network
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks
    
class MaskMambaBlock(nn.Module):
    def __init__(
        self,
        n_embd,  # dimension of the input features
        kernel_size=4,  # conv kernel size
        n_ds_stride=1,  # downsampling stride for the current layer
        drop_path_rate=0.3,  # drop path rate
        d_state=64,  # dimension of the state
        init_conv_vars=1,
        use_mamba_type="dbm",
    ):
        super().__init__()
        if use_mamba_type == "dbm":
            self.mamba = DBM(n_embd, d_conv=kernel_size, use_fast_path=True, expand=1)
            # self.chunk_mamba = DBM(int(n_embd / 2), d_conv=kernel_size, use_fast_path=True, expand=1)
        elif use_mamba_type == "vim":
            # vim
            self.mamba = ViM(n_embd, d_conv=kernel_size, bimamba_type="v2", use_fast_path=True)
        elif use_mamba_type == "mamba2":
            self.mamba = Mamba2(n_embd, d_conv=kernel_size, use_mem_eff_path=True, expand=1, d_state=d_state)
        else:
            raise NotImplementedError
        if n_ds_stride > 1:
            self.downsample = MaxPooler(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None

        self.norm = nn.LayerNorm(n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.global_fc.bias, 0)


        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        res = x
        x_ = x.transpose(1, 2)      #x: 2 512 2304
        x_ = self.norm(x_)
        gf = self.mamba(x_).transpose(1, 2)

        phi = torch.relu(self.global_fc(x_.transpose(1,2).mean(dim=-1, keepdim=True)))
        
        # x, z = x_.chunk(2, dim=-1)
        # y = self.chunk_mamba(x).transpose(1, 2)
        # y = y * torch.relu(z.transpose(1, 2))

        # yy = self.chunk_mamba(z).transpose(1, 2)
        # yy = yy * torch.relu(x.transpose(1, 2))
        # y = torch.cat([y, yy], dim=1)

        out = gf * phi + x_.transpose(1, 2)


        x = out * mask.unsqueeze(1).to(x.dtype)

        x = res + self.drop_path(x)

        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        return x, mask
    
class MaxPooler(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)

        self.stride = stride

    def forward(self, x, mask, **kwargs):
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = self.ds_pooling(mask.float()).bool()
        else:
            # masking out the features
            out_mask = mask

        out = self.ds_pooling(x) * out_mask.unsqueeze(1).to(x.dtype)

        return out, out_mask.bool()