import torch
from ...builder import PRIOR_GENERATORS
import torch.nn as nn
import torch.nn.functional as F


@PRIOR_GENERATORS.register_module()
class PointGenerator:
    def __init__(
        self,
        strides,  # strides of fpn levels
        regression_range,  # regression range (on feature grids)
        use_offset=False,  # if to align the points at grid centers
    ):
        super().__init__()
        self.strides = strides
        self.regression_range = regression_range
        self.use_offset = use_offset

    def __call__(self, feat_list):
        # feat_list: list[B,C,T]

        pts_list = []
        for i, feat in enumerate(feat_list):
            T = feat.shape[-1]

            points = torch.linspace(0, T - 1, T, dtype=torch.float) * self.strides[i]  # [T]
            reg_range = torch.as_tensor(self.regression_range[i], dtype=torch.float)
            stride = torch.as_tensor(self.strides[i], dtype=torch.float)

            if self.use_offset:
                points += 0.5 * stride

            points = points[:, None]  # [T,1]
            reg_range = reg_range[None].repeat(T, 1)  # [T,2]
            stride = stride[None].repeat(T, 1)  # [T,1]
            pts_list.append(torch.cat((points, reg_range, stride), dim=1).to(feat.device))  # [T,4]
        return pts_list
@PRIOR_GENERATORS.register_module()
class AdaptivePointGenerator(nn.Module): 
    def __init__(self, scale_head=512, use_offset=False, strides=None, regression_range=None):
        super().__init__()
        self.use_offset = use_offset
        self.num_levels = len(strides)
        assert self.num_levels == len(regression_range)
        # 用于预测每层的动作长度 scale
        self.scale_head = nn.Linear(scale_head, 1)  # 假设输入 C=
        self.strides = strides

        # 可学习的 stride（log空间保证正数 & 单调递增）
        log_strides = torch.log(torch.tensor(strides, dtype=torch.float))
        self.log_strides = nn.Parameter(log_strides)
 
        # 可学习的 regression range 上下界（保持单调递增）
        reg_mins = torch.tensor([r[0] for r in regression_range], dtype=torch.float)
        reg_maxs = torch.tensor([r[1] for r in regression_range], dtype=torch.float)
        # print(reg_mins)
        # print(reg_maxs)
 
        self.raw_mins = nn.Parameter(reg_mins)
        self.raw_maxs = nn.Parameter(reg_maxs)
    
    def forward(self, feat_list):
        pts_list = []
        strides = torch.exp(self.log_strides)  # 保证正数
 
        # 用 softplus + cumsum 保证递增
        reg_min = torch.cumsum(F.softplus(self.raw_mins), dim=0)
        reg_max = torch.cumsum(F.softplus(self.raw_maxs), dim=0)
 
        for i, feat in enumerate(feat_list):
            B, C, T = feat.shape
            device = feat.device
 
            stride = strides[i].to(device)
            points = torch.arange(0, T, dtype=torch.float, device=device) * stride
 
            if self.use_offset:
                points += 0.5 * stride
 
            points = points[:, None]  # [T,1]
            reg_range = torch.cat([
                reg_min[i].expand(T, 1),
                reg_max[i].expand(T, 1)
            ], dim=1)  # [T,2]
            stride_col = stride.expand(T, 1)
 
            pts_list.append(torch.cat((points, reg_range, stride_col), dim=1))  # [T,4]
            # print(pts_list)
 
        return pts_list