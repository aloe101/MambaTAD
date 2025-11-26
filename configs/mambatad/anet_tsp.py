_base_ = [
    "../_base_/datasets/activitynet-1.3/features_tsp_resize_trunc.py",  # dataset config
    "../_base_/models/videomambasuite.py",  # model config
]
data_path = "/home/Dataset/DatasetHui/tad/anet_tsp_npy_unresize"
model = dict(
    projection=dict(
        in_channels=512,
        out_channels=256,
        attn_cfg=dict(n_mha_win_size=[7, 7, 7, 7, 7, -1]),
        use_abs_pe=True,
        max_seq_len=192,
        input_pdrop=0.2,
        mamba_cfg=dict(kernel_size=4, drop_path_rate=0.1, use_mamba_type="bssdm"),
        use_global=True,
    ),
    neck=dict(in_channels=256, out_channels=256),
    rpn_head=dict(
        in_channels=256,
        feat_channels=256,
        num_classes=1,
        label_smoothing=0.1,
        loss_weight=2.0,
        loss_normalizer=200,
    ),
)

solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=20)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.75,
        max_seg_num=100,
        iou_threshold=0,  # does not matter when use soft nms
        min_score=0.001,
        multiclass=False,
        voting_thresh=0.9,  #  set 0 to disable
    ),
    external_cls=dict(
        type="StandardClassifier",
        path="data/activitynet-1.3/classifiers/anet_UMTv2_6B_k710+K40_f16_frozenTuning.json_converted.json",
        topk=2,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=0    #7,
)

work_dir = "exps/anet/bssdm_tsp"
