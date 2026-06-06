
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import datetime
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from mmengine.optim.scheduler.lr_scheduler import PolyLR
from timm.utils import ModelEmaV2

import util.misc as utils
from engine import train_one_epoch
from models import build_model
from datasets import create_dataset
from eval.evaluate import eval
from util.logger import get_logger


# =====================================================================
# 🌟 ABL (Active Boundary Loss)
# =====================================================================
class ActiveBoundaryLoss(nn.Module):
    def __init__(self, base_criterion, lambda_weight=0.05):
        super().__init__()
        self.base_criterion = base_criterion
        self.lambda_weight = lambda_weight

        sobel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]]
        ).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def get_boundary(self, x):
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return edge

    def forward(self, pred, target):
        base_loss = self.base_criterion(pred, target.float())
        pred_prob = torch.sigmoid(pred)
        pred_boundary = self.get_boundary(pred_prob)
        target_boundary = self.get_boundary(target.float())
        boundary_loss = F.l1_loss(pred_boundary, target_boundary)
        return base_loss + self.lambda_weight * boundary_loss
# =====================================================================


def get_args_parser():
    parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', add_help=False)

    parser.add_argument('--BCELoss_ratio', default=0.83, type=float)
    parser.add_argument('--DiceLoss_ratio', default=0.17, type=float)
    parser.add_argument('--Norm_Type', default='GN', type=str)

    parser.add_argument('--dataset_path', default="../data/DeepCrack")

    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--batch_size_test', type=int, default=1)

    parser.add_argument('--lr_scheduler', type=str, default='PolyLR')
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--sgd', action='store_true')

    parser.add_argument('--output_dir', default='./checkpoints/DeepCrack_Full_Model')

    # ==============================
    # 🌟 断点续训参数
    # ==============================
    parser.add_argument('--resume_ckpt', default='', type=str)

    # 兼容“旧 checkpoint 没保存 best 指标”的情况
    parser.add_argument('--resume_best_epoch', default=None, type=int)
    parser.add_argument('--resume_best_miou', default=None, type=float)
    parser.add_argument('--resume_best_ods', default=None, type=float)
    parser.add_argument('--resume_best_ois', default=None, type=float)
    parser.add_argument('--resume_best_f1', default=None, type=float)
    parser.add_argument('--resume_best_precision', default=None, type=float)
    parser.add_argument('--resume_best_recall', default=None, type=float)

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_mode', type=str, default='crack')
    parser.add_argument('--serial_batches', action='store_true')
    parser.add_argument('--num_threads', default=1, type=int)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--load_width', type=int, default=512)
    parser.add_argument('--load_height', type=int, default=512)
    return parser


def move_optimizer_state_to_device(optimizer, device):
    """把 optimizer state 里的 tensor 移到目标设备，避免 resume 后设备不一致。"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def save_training_state(output_path, model, ema_model, optimizer, lr_scheduler, epoch, args,
                        max_miou, max_metrics, run_name):
    utils.save_on_master({
        'model': model.state_dict(),
        'ema_model': ema_model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
        'max_mIoU': max_miou,
        'max_Metrics': max_metrics,
        'run_name': run_name,
    }, output_path)


def main(args):
    checkpoints_path = "./checkpoints"
    cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
    dataset_name = Path(args.dataset_path).name

    # ==============================
    # 🌟 实验名：新跑用当前时间戳；续训则沿用旧实验目录名
    # ==============================
    run_name = f'{cur_time}_Dataset->{dataset_name}'
    if args.resume_ckpt:
        run_name = Path(args.resume_ckpt).parent.name

    process_folder_path = os.path.join(checkpoints_path, run_name)
    os.makedirs(process_folder_path, exist_ok=True)

    log_train = get_logger(process_folder_path, 'train')
    log_test = get_logger(process_folder_path, 'test')
    log_eval = get_logger(process_folder_path, 'eval')

    device = torch.device(args.device)
    args.phase = 'train'

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, base_criterion = build_model(args)
    criterion = ActiveBoundaryLoss(base_criterion, lambda_weight=0.05)

    model.to(device)
    criterion.to(device)

    ema_model = ModelEmaV2(model, decay=0.999)

    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)

    param_dicts = [{"params": [p for _, p in model.named_parameters()], "lr": args.lr}]

    if args.sgd:
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )

    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-5
        )
    elif args.lr_scheduler == 'PolyLR':
        lr_scheduler = PolyLR(
            optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")

    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================
    # 🌟 best 记录
    # ==============================
    max_mIoU = 0.0
    max_Metrics = {
        'epoch': 0,
        'mIoU': 0.0,
        'ODS': 0.0,
        'OIS': 0.0,
        'F1': 0.0,
        'Precision': 0.0,
        'Recall': 0.0
    }

    # ==============================
    # 🌟 断点续训
    # ==============================
    if args.resume_ckpt:
        resume_ckpt = Path(args.resume_ckpt)
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt}")

        print(f"\n⚠️ Resume from: {resume_ckpt}\n")
        log_train.info(f"Resume from: {resume_ckpt}")

        checkpoint = torch.load(resume_ckpt, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        ema_model.module.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        move_optimizer_state_to_device(optimizer, device)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        args.start_epoch = checkpoint['epoch'] + 1

        if 'max_mIoU' in checkpoint:
            max_mIoU = checkpoint['max_mIoU']
        if 'max_Metrics' in checkpoint:
            max_Metrics = checkpoint['max_Metrics']

        # 兼容旧 checkpoint：如果其中没有 best 记录，就用命令行手动补
        if max_mIoU == 0.0 and args.resume_best_miou is not None:
            max_mIoU = args.resume_best_miou
            max_Metrics = {
                'epoch': args.resume_best_epoch if args.resume_best_epoch is not None else 0,
                'mIoU': args.resume_best_miou if args.resume_best_miou is not None else 0.0,
                'ODS': args.resume_best_ods if args.resume_best_ods is not None else 0.0,
                'OIS': args.resume_best_ois if args.resume_best_ois is not None else 0.0,
                'F1': args.resume_best_f1 if args.resume_best_f1 is not None else 0.0,
                'Precision': args.resume_best_precision if args.resume_best_precision is not None else 0.0,
                'Recall': args.resume_best_recall if args.resume_best_recall is not None else 0.0,
            }
            print(f"✅ Inject old best metrics from CLI, best epoch = {max_Metrics['epoch']}")
            log_train.info(f"Inject old best metrics from CLI, best epoch = {max_Metrics['epoch']}")

        print(f"✅ Resume success. Start from epoch {args.start_epoch}")
        log_train.info(f"Resume success. Start from epoch {args.start_epoch}")

        if max_mIoU == 0.0:
            warn_msg = (
                "WARNING: resumed from an old checkpoint without stored best metrics. "
                "checkpoint_best.pth may be overwritten by later epochs. "
                "Use --resume_best_* to preserve previous best."
            )
            print(warn_msg)
            log_train.info(warn_msg)

    log_train.info("effective args -> " + str(args))

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        args.phase = 'train'
        args.batch_size = args.batch_size_train

        print(f"training epoch start -> {epoch}")
        train_one_epoch(
            model, criterion, train_dataLoader, optimizer, epoch,
            args=args, logger=log_train, ema_model=ema_model
        )
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            checkpoint_paths.append(output_dir / f'checkpoint{epoch}.pth')

            for checkpoint_path in checkpoint_paths:
                save_training_state(
                    checkpoint_path, model, ema_model, optimizer, lr_scheduler,
                    epoch, args, max_mIoU, max_Metrics, run_name
                )

        print(f"testing epoch start -> {epoch}")
        results_path = run_name
        save_root = f'./results/{results_path}/results_{epoch}'

        args.phase = 'test'
        args.batch_size = args.batch_size_test
        test_dl = create_dataset(args)
        pbar = tqdm(total=len(test_dl), desc="Initial Loss: Pending")

        os.makedirs(save_root, exist_ok=True)

        with torch.no_grad():
            ema_model.module.eval()
            for _, data in enumerate(test_dl):
                x = data["image"].to(device)
                target = data["label"].float().to(device)

                out = ema_model.module(x)

                pred_prob = torch.sigmoid(out[0, 0, ...]).detach().cpu().numpy()
                out_img = (pred_prob * 255.0).clip(0, 255).astype(np.uint8)

                target_np = target[0, 0, ...].detach().cpu().numpy()
                target_img = (target_np * 255.0).clip(0, 255).astype(np.uint8)

                root_name = data["A_paths"][0].split("/")[-1][0:-4]

                cv2.imwrite(os.path.join(save_root, f"{root_name}_lab.png"), target_img)
                cv2.imwrite(os.path.join(save_root, f"{root_name}_pre.png"), out_img)
                pbar.update(1)
        pbar.close()

        metrics = eval(log_eval, save_root, epoch)
        for key, value in metrics.items():
            print(f"{key} -> {value}")

        if max_mIoU < metrics['mIoU']:
            max_Metrics = dict(metrics)
            max_Metrics['epoch'] = epoch
            max_mIoU = metrics['mIoU']

            save_training_state(
                output_dir / 'checkpoint_best.pth',
                model, ema_model, optimizer, lr_scheduler,
                epoch, args, max_mIoU, max_Metrics, run_name
            )

    total_time = time.time() - start_time
    print(f'Process time {str(datetime.timedelta(seconds=int(total_time)))}')
    log_train.info(f'Process time {str(datetime.timedelta(seconds=int(total_time)))}')

    print("\n" + "=" * 50)
    print(f"🎉 Training Complete! Best Metrics at Epoch {max_Metrics['epoch']}:")
    for k, v in max_Metrics.items():
        if k != 'epoch':
            print(f"{k}: {v:.4f}")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)