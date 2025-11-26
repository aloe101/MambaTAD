_base_ = ["e2e_anet_videomae_s_192x4_160_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            patch_size=14,
            embed_dims=1408,
            depth=40,
            num_heads=16,
            mlp_ratio=48 / 11,
            adapter_index=list(range(40)),
        ),
        custom=dict(pretrain="pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth"),
    ),
    projection=dict(in_channels=1408),
)

solver = dict(
    train=dict(batch_size=4, num_workers=4),
    val=dict(batch_size=4, num_workers=4),
    test=dict(batch_size=4, num_workers=4),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

work_dir = "exps/anet/adatad/e2e_actionformer_videomaev2_g_192x4_160_adapter"
