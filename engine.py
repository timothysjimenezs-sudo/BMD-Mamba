
from typing import Iterable
import torch
import time
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, args=None, logger=None, ema_model=None):
    model.train()
    criterion.train()

    accumulation_steps = getattr(args, 'accum_steps', 4)
    optimizer.zero_grad()

    total_iters = len(data_loader)
    pbar = tqdm(total=total_iters, desc="Initial Loss Fused: Pending")

    for i, data in enumerate(data_loader):
        samples = data['image'].to(torch.device(args.device))
        targets = data['label'].to(torch.device(args.device))

        output = model(samples)

        # 真实 loss（用于显示和日志）
        loss_final = criterion(output, targets.float())

        # 处理最后一个不足 accumulation_steps 的累积组
        remainder = total_iters % accumulation_steps
        is_last_group = (remainder != 0 and i >= total_iters - remainder)
        current_accum_steps = remainder if is_last_group else accumulation_steps

        # 反向传播用的缩放 loss
        loss_for_backward = loss_final / current_accum_steps
        loss_for_backward.backward()

        cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
        loss_final_str = '{:.4f}'.format(loss_final.item())
        l = optimizer.param_groups[0]['lr']
        logger.info(
            f"time -> {cur_time} | Epoch -> {epoch} | image_num -> {data['A_paths']} | "
            f"loss final -> {loss_final_str} | lr -> {l}"
        )

        pbar.set_description(f"Loss: {loss_final.item():.4f}")
        pbar.update(1)

        # 到达累积步数，或者已经是最后一个 batch，执行一次参数更新
        if (i + 1) % accumulation_steps == 0 or (i + 1 == total_iters):
            optimizer.step()

            if ema_model is not None:
                ema_model.update(model)

            optimizer.zero_grad()

    pbar.close()