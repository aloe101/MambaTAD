import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .actionformer_proj import get_sinusoid_encoding
from ..bricks import ConvModule, AffineDropPath, TransformerBlock, SGPBlock
from ..builder import PROJECTIONS

try:
    from mamba_ssm.modules.mamba_simple import Mamba as ViM
    from mamba_ssm.modules.mamba_new import Mamba as DBM
    from mamba_ssm.modules.bssdm import Mamba2
    from mamba_ssm.modules.hydra import Hydra
    from mamba_ssm.modules.mod_mamba import Mamba as FABS6

    MAMBA_AVAILABLE = True

except ImportError as e:
    print(e)
    MAMBA_AVAILABLE = False



@PROJECTIONS.register_module()
class S6MambaProj(nn.Module):
    """Implementation of Video-Mamba-Suite: https://arxiv.org/abs/2403.09626"""

    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        conv_cfg=None,  # kernel_size proj_pdrop
        norm_cfg=None,
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        input_pdrop=0.0,  # drop out the input feature
        input_noise=0.0,
        use_stem=True,
        init_conv_vars=1,  # initialization of gaussian variance for the weight in SGP
        mamba_cfg=dict(kernel_size=4, drop_path_rate=0.3, use_mamba_type="dbm"),  # default to DBM
        h_hidden=768,
        use_global=False,
        drop_path_rate_out=0.1,
        former=False,
        sgp=False,
        sgp_mlp_dim=768,
        k=1.5,
        downsample_type="max",
        sgp_win_size=[1, 1, 1, 1, 1, 1],
        attn_cfg=dict(n_head=4, n_mha_win_size=19),
        path_pdrop=0.1,
    ):
        super().__init__()
        assert (
            MAMBA_AVAILABLE
        ), "Please install mamba-ssm to use this module. Check: https://github.com/OpenGVLab/video-mamba-suite"

        assert len(arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.scale_factor = 2  # as default
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len
        self.input_noise = input_noise
        self.use_stem = use_stem
        self.use_global = use_global
        self.sgp = sgp
        self.former = former
        self.n_mha_win_size = attn_cfg["n_mha_win_size"]
        self.n_head = attn_cfg["n_head"]
        self.path_pdrop = path_pdrop
        self.attn_pdrop = 0.0
        self.proj_pdrop = 0.0
        self.n_mha_win_size = attn_cfg["n_mha_win_size"]
        self.sgp_win_size = sgp_win_size
        if isinstance(self.n_mha_win_size, int):
            self.mha_win_size = [self.n_mha_win_size] * (1 + arch[-1])
        else:
            assert len(self.n_mha_win_size) == (1 + arch[-1])
            self.mha_win_size = self.n_mha_win_size

        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None

        if isinstance(self.in_channels, (list, tuple)):
            assert isinstance(self.out_channels, (list, tuple)) and len(self.in_channels) == len(self.out_channels)
            self.proj = nn.ModuleList([])
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj.append(
                    ConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj = None

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

        # stem network using (vanilla) transformer
        if self.use_stem:
            # self.stem = nn.ModuleList()
            self.blocks = nn.ModuleList()
            for _ in range(arch[1]):
                if self.former:
                    self.stem.append(
                    TransformerBlock(
                        out_channels,
                        self.n_head,
                        n_ds_strides=(1, 1),
                        attn_pdrop=self.attn_pdrop,
                        proj_pdrop=self.proj_pdrop,
                        path_pdrop=self.path_pdrop,
                        mha_win_size=self.mha_win_size[0],
                    )
                    )
                elif self.sgp:
                    self.stem.append(
                SGPBlock(
                    out_channels,
                    kernel_size=1,
                    n_ds_stride=1,
                    n_hidden=sgp_mlp_dim,
                    k=k,
                    init_conv_vars=init_conv_vars,
                    # mamba=mamba,
                ))
                else:
                    # self.stem.append(MaskMambaBlock(out_channels, **mamba_cfg))
                    self.recurrent = 4
                    ln = nn.LayerNorm(out_channels)
                    t_mamba = FABS6(d_model=out_channels, 
                                    d_conv=4, 
                                    use_fast_path=True, 
                                    expand=1, 
                                    d_state=16, 
                                    num_kernels=3)
                    c_mamba = FABS6(d_model=2304//4, 
                                    d_conv=4, 
                                    use_fast_path=True, 
                                    expand=1, 
                                    d_state=16, 
                                    num_kernels=3)
                    drop_path = AffineDropPath(num_dim=out_channels, 
                                            drop_prob=0.3)
                    adaptive_pool = nn.AdaptiveAvgPool1d(output_size=2304//4)
                    final_pool = nn.AvgPool1d(kernel_size=2304//4)
                    
                    self.blocks.append(nn.ModuleDict({
                        'ln': ln,
                        't_mamba': t_mamba,
                        'c_mamba': c_mamba,
                        'drop_path': drop_path,
                        'adaptive_pool': adaptive_pool,
                        'final_pool': final_pool,
                        'sigmoid': nn.Sigmoid()
                    }))
        torch.use_deterministic_algorithms(True, warn_only=True)


        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            if self.former:
                    self.branch.append(
                    TransformerBlock(
                        out_channels,
                        self.n_head,
                        n_ds_strides=(self.scale_factor, self.scale_factor),
                        attn_pdrop=self.attn_pdrop,
                        proj_pdrop=self.proj_pdrop,
                        path_pdrop=self.path_pdrop,
                        mha_win_size=self.mha_win_size[1+idx],
                    )
                    )
            elif self.sgp:
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
                    # mamba=mamba,
                ))
            else:
                # self.branch.append(MaskMambaBlock(out_channels, init_conv_vars=init_conv_vars, n_ds_stride=2, **mamba_cfg))
                ln = nn.LayerNorm(out_channels)
                mamba = FABS6(d_model=out_channels, 
                            d_conv=4, 
                            use_fast_path=True, 
                            expand=1, 
                            d_state=16)
                drop_path = AffineDropPath(num_dim=out_channels, drop_prob=0.3)
                ds_maxpool = MdMaxPooler(kernel_size=3, stride=2, padding=1)  # l -> 0.5l
                
                self.branch.append(nn.ModuleDict({
                    'mamba': mamba,
                    'ln': ln,
                    'drop_path': drop_path,
                    'ds_maxpool': ds_maxpool
                }))

        self.linear = nn.Sequential(
                nn.Linear(out_channels, h_hidden),
                nn.ReLU(),
                nn.Linear(h_hidden, out_channels),
            )
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.upconv = nn.Conv1d(out_channels*2, out_channels, 1)

        self.global_mamba = MaskMambaBlock(out_channels, **mamba_cfg)
        if drop_path_rate_out > 0.0:
            self.drop_path = AffineDropPath(out_channels, drop_prob=drop_path_rate_out)
        else:
            self.drop_path = nn.Identity()

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)
        

        # if self.input_noise > 0:
        #     noise = torch.randn_like(x) * self.input_noise
        #     x += noise

        # feature projection
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        # drop out input if needed
        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

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
        # if self.use_stem:
        # for idx in range(len(self.stem)):
        #     x, mask = self.stem[idx](x, mask)
        mask = mask.unsqueeze(1)
        for block in self.blocks:
            res = x
            for _ in range(self.recurrent):
                x_t = rearrange(x, 'b c l -> b l c')
                x_t = block['ln'](x_t)
                x_t = block['t_mamba'](x_t)
                x_t = rearrange(x_t, 'b l c -> b c l')
                x_t = x_t * mask.to(x.dtype)
                
                with torch.backends.cudnn.flags(enabled=False):
                    x_c = block['adaptive_pool'](x)  # b c l -> b c l//4
                x_c = block['c_mamba'](x_c)
                x_c = block['final_pool'](x_c)  # b c l//4 -> b c 1 using AvgPool1d
                x_c = block['sigmoid'](x_c)  # Apply sigmoid after c_mamba
                x_c = F.interpolate(x_c, size=x_t.size(-1), mode='nearest')  # b c l//4 -> b c l
                x_c = x_c * mask.to(x.dtype)
                
                x = x_t * x_c  # SE block mechanism
                x = block['drop_path'](x)
                x = x + res
        # return x, mask

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask.squeeze(1),)
        # final_feats = (x,)
        # final_masks = (mask,)

        # main branch with downsampling
        # for idx in range(len(self.branch)):
        #     x, mask = self.branch[idx](x, mask)
        #     out_feats += (x,)
        #     out_masks += (mask,)

            # if idx < 3:
            #     final_feats += (x,)
            #     final_masks += (mask,)
        
        for block in self.branch:
            res = x
            x = rearrange(x, 'b c l -> b l c')
            x = block['ln'](x)
            x = block['mamba'](x)
            x = rearrange(x, 'b l c -> b c l')
            x = x * mask.to(x.dtype)
            x = block['drop_path'](x)
            x = x + res
            x, mask = block['ds_maxpool'](x, mask)
            out_feats += (x, )
            out_masks += (mask.squeeze(1),)
        
        # return out_feats, out_masks

        # x_up1 = self.linear(x.transpose(1,2)).transpose(1,2)
        
        # x_up2= self.up(x_up1)
        # x2 = out_feats[-2]
        # pad_mask = out_masks[-2]
        # if x_up2.shape[2] > x2.shape[2]:
        #     x_pre = out_feats[-2]
        #     x2 = F.pad(x_pre, (0, x_up2.shape[2]-x_pre.shape[2]))
        #     pad_mask = torch.zeros((1, x2.shape[2]), dtype=torch.bool, device=x2.device)
        #     pad_mask[:,:x_pre.shape[2]] = out_masks[-2]
        # x_up2 = torch.cat([x2, x_up2], dim=1)
        # # mask_up2 = torch.cat([mask_pre, mask_up], dim=1)
        
        # x_up2 = self.upconv(x_up2)
        # final_feats += (x_up2,)
        # final_feats += (x_up1,)
        # final_masks += (pad_mask,)
        # final_masks += (mask,)
        if self.use_global:
            x = torch.cat(out_feats, dim=-1)
            m = torch.cat(out_masks, dim=-1)
            res = x
            x, m = self.global_mamba(x, m)
            x = res + self.drop_path(x)
            split_size = [mask.shape[-1] for mask in out_masks]
            out_feats = torch.split(x, split_size, dim=-1)
            out_masks = torch.split(m, split_size, dim=-1)

        return out_feats, out_masks


class MaskMambaBlock(nn.Module):
    def __init__(
        self,
        n_embd,  # dimension of the input features
        kernel_size=4,  # conv kernel size
        n_ds_stride=1,  # downsampling stride for the current layer
        drop_path_rate=0.3,  # drop path rate
        d_state=64,  # dimension of the state
        d_conv=7,  # dimension of the conv
        use_mamba_type="dbm",
        init_conv_vars=1,
        expand=1,
    ):
        super().__init__()
        if use_mamba_type == "dbm":
            self.mamba = DBM(n_embd, d_conv=kernel_size, use_fast_path=True, expand=1)
        elif use_mamba_type == "vim":
            # vim
            self.mamba = ViM(n_embd, d_conv=kernel_size, bimamba_type="v2", use_fast_path=True)
        elif use_mamba_type == "mamba2":
            self.mamba = Mamba2(n_embd, d_conv=kernel_size, use_mem_eff_path=True, expand=1, d_state=d_state)
        elif use_mamba_type == "hydra":
            self.mamba = Hydra(n_embd, d_conv=d_conv, use_mem_eff_path=True, expand=expand, d_state=d_state)
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
        x_ = self.mamba(x_).transpose(1, 2)
        # phi = torch.relu(self.global_fc(x_.transpose(1,2).mean(dim=-1, keepdim=True)))
        # out = gf * phi + x_.transpose(1, 2)
        x = x_ * mask.unsqueeze(1).to(x.dtype)

        x = res + self.drop_path(x)

        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        return x, mask

class MdMaxPooler(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        self.stride = stride
    
    def forward(self, x, mask, **kwargs):
        # out_mask = F.interpolate(mask.to(x.dtype), size=x.size(-1)//self.stride, mode='nearest') if self.stride > 1 else mask
        out_mask = self.ds_pooling(mask.float()).bool() if self.stride > 1 else mask
        out = self.ds_pooling(x)*out_mask.to(x.dtype)
        return out, out_mask.bool()

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
