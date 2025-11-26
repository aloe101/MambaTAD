_base_ = [
    "../_base_/datasets/multithumos/features_i3d_rgb_pad.py",  # dataset config
    "../_base_/models/videomambasuite.py",  # model config
]

# trunc_len = 2304
data_path = "/home/Dataset/DatasetHui/tad/i3d_actionformer_stride4_thumos/i3d_actionformer_stride4_thumos/"


model = dict(
    projection=dict(in_channels=1024, arch=(3, 0, 5), input_pdrop=0.1,
                    mamba_cfg=dict(kernel_size=4, drop_path_rate=0.3, use_mamba_type="bssdm"),use_global=True,),
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

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=65)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    pre_nms_topk=8000,
    pre_nms_thresh=0.001,
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=8000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=39,
    end_epoch=55,
)

work_dir = "exps/multithumos/bssdm_i3d_rgb"
