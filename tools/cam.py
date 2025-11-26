import os
import cv2
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)
import numpy as np
import argparse
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import train_one_epoch, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from opentad.models.backbones.backbone_wrapper import BackboneWrapper
from opentad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)

from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("--config", type=str, default='/home/hui007/tad/OpenTAD/configs/adatad/thumos/e2e_thumos_videomae_l_768x1_160_adapter_mamba_1.py', help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--checkpoint", type=str, default="/home/hui007/tad/OpenTAD/pretrained/epoch_59_bssdm_adp.pth", help="resume from a checkpoint")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")

    # parser.add_argument('--device', type=str, default='cpu',
    #                     help='Torch device to use')
    parser.add_argument(
        '--image-path',
        type=str,
        default='/home/hui007/tad/frame__1050.jpg',
        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')
    args = parser.parse_args()
    
    # if args.device:
    #     print(f'Using device "{args.device}" for acceleration')
    # else:
    #     print('Using CPU for computation')

    return args



def main():
    args = parse_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM
    }

    # load config
    cfg = Config.fromfile(args.config)

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir
    set_seed(args.seed)
    cfg = update_workdir(cfg, args.id, torch.cuda.device_count())
    if args.rank == 0:
        create_folder(cfg.work_dir)

    # setup logger
    # logger = setup_logger("Test", save_dir=cfg.work_dir, distributed_rank=args.rank)
    # logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    # logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    # test_dataset = build_dataset(cfg.dataset.test)
    # test_loader = build_dataloader(
    #     test_dataset,
    #     rank=args.rank,
    #     world_size=args.world_size,
    #     shuffle=False,
    #     drop_last=False,
    #     **cfg.solver.test,
    # )

    # build model
    # model = build_detector(cfg.model)
    model = BackboneWrapper(cfg.model["backbone"])

    # DDP
    model = model.to(args.local_rank)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # logger.info(f"Using DDP with total {args.world_size} GPUS...")

    # load checkpoint: args -> config -> best
    if args.checkpoint != "none":
        checkpoint_path = args.checkpoint
    elif "test_epoch" in cfg.inference.keys():
        checkpoint_path = os.path.join(cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth")
    else:
        checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")
    print("Loading checkpoint from: {}".format(checkpoint_path))
    device = f"cuda:{args.rank % torch.cuda.device_count()}"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Checkpoint is epoch {}.".format(checkpoint["epoch"]))

    # Model EMA
    print(checkpoint.keys())    #dict_keys(['epoch', 'model', 'model_ema', 'optimizer', 'scheduler', 'mAP'])
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        # print(checkpoint["state_dict_ema"].skeys())
        model.load_state_dict(checkpoint["state_dict_ema"], strict=False)
        # model.load_state_dict(checkpoint["model_ema"])
        # logger.info("Using Model EMA...")
    else:
        model.load_state_dict(checkpoint["state_dict"])

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        print("Using Automatic Mixed Precision...")

    # test the detector
    print("Testing Starts...\n")
    # eval_one_epoch(
    #     test_loader,
    #     model,
    #     cfg,
    #     logger,
    #     args.rank,
    #     model_ema=None,  # since we have loaded the ema model above
    #     use_amp=use_amp,
    #     world_size=args.world_size,
    #     not_eval=args.not_eval,
    # )
    model = model.eval()
    # for attr in model.module.backbone.named_modules():
    #     print(attr)
    # print("\n")
    # for attr in dir(model.module.backbone.block):
    #     print(attr)
    for name, module in model.module.named_modules():
        if name == 'model.backbone.fc_norm':
            target_layers = module
            break
    target_layers = [target_layers]
    print(target_layers)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(device)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).transpose(2,3).expand(-1, -1,-1, 768,-1,-1)
    targets = None
    cam_algorithm = methods[args.method]
    try:
        with cam_algorithm(model=model,
                        target_layers=target_layers) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)
            print("grayscale_cam type:", type(grayscale_cam))

            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"error: {e}")

    gb_model = GuidedBackpropReLUModel(model=model, device=device)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    os.makedirs(args.output_dir, exist_ok=True)

    cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam.jpg')
    gb_output_path = os.path.join(args.output_dir, f'{args.method}_gb.jpg')
    cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_cam_gb.jpg')

    cv2.imwrite(cam_output_path, cam_image)
    cv2.imwrite(gb_output_path, gb)
    cv2.imwrite(cam_gb_output_path, cam_gb)
    
    print("Testing Over...\n")


if __name__ == "__main__":
    main()
