_base_ = ["e2e_anet_videomae_s_192x4_160_adapter_mamba.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1280,
            depth=32,
            num_heads=16,
            adapter_index=list(range(32)),
            # use_mamba_adapter=True,
            # drop_path_rate_out = 0.2,
            # mamba_cfg=dict(kernel_size=4, drop_path_rate=0.1, use_mamba_type="dbm"),
        ),
        custom=dict(pretrain="pretrained/vit-huge-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"),
    ),
    projection=dict(in_channels=1280),
)

solver = dict(
    train=dict(batch_size=4, num_workers=4),
    val=dict(batch_size=4, num_workers=4),
    test=dict(batch_size=4, num_workers=4),
)

optimizer = dict(
    lr=1e-4,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=1e-5, weight_decay=0.05)],
        exclude=["backbone"],
    )
)

workflow = dict(
    val_start_epoch=0,
)

work_dir = "exps/anet/adatad/e2e_actionformer_videomae_h_192x4_160_adapter_mamba"
