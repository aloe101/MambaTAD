_base_ = ["e2e_bssdm_s.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1024,
            depth=24,
            num_heads=16,
            adapter_index=list(range(24)),
            drop_path_rate_out = 0.3,
            mamba_type="bssdm",
            use_mamba_adapter=True,
            mamba_cfg=dict(kernel_size=4, drop_path_rate=0.2, use_mamba_type="bssdm"),
        ),
        custom=dict(pretrain="pretrained/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth"),
    ),
    projection=dict(in_channels=1024, mamba_cfg=dict(kernel_size=4, drop_path_rate=0.3, use_mamba_type="bssdm"),
                     input_pdrop=0.1, drop_path_rate_out=0.2,
        use_global=True,),
        rpn_head=dict(
        type="TriDetHead",
        num_classes=65,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        boundary_kernel_size=3,
        iou_weight_power=0.2,
        num_bins=16,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
            iou_rate=dict(type="GIOULoss"),
        ),
        # mse_loss=True,
    ),
)

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=1e-5, weight_decay=0.05)]))

work_dir = "exps/multithumos/adatad/e2e_bssdm_l_adapter"