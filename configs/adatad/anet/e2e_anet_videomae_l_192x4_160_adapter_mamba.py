_base_ = ["e2e_anet_videomae_s_192x4_160_adapter_mamba.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1024,
            depth=24,
            num_heads=16,
            adapter_index=list(range(24)),
        ),
        custom=dict(pretrain="pretrained/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth"),
    ),
    projection=dict(in_channels=1024,mamba_cfg=dict(kernel_size=4, drop_path_rate=0.1, use_mamba_type="bssdm"),
    use_global=True,input_pdrop=0.2, drop_path_rate_out=0.3,),
)

solver = dict(
    train=dict(batch_size=4, num_workers=4),
    val=dict(batch_size=4, num_workers=4),
    test=dict(batch_size=4, num_workers=4),
)

optimizer = dict(
    lr=1e-3,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=1e-5, weight_decay=0.05)],
        exclude=["backbone"],
    )
)
workflow = dict(
    val_start_epoch=2,
)

work_dir = "exps/anet/adatad/e2e_actionformer_videomae_l_192x4_160_adapter_mamba"
