"""
Train a noised image regressor to predict compliance.
"""

import argparse
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from topodiff import dist_util, logger
from topodiff.fp16_util import MixedPrecisionTrainer
from topodiff.image_datasets_regressor import load_data
from topodiff.resample import create_named_schedule_sampler
from topodiff.script_util_IH import (
    add_dict_to_argparser,
    args_to_dict,
    regressor_and_diffusion_defaults,
    create_regressor_and_diffusion,
    create_regressorMuRho_and_diffusion,
    create_regressorLinear_and_diffusion
)
from topodiff.train_util_IH import parse_resume_step_from_filename, log_loss_dict
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()


    logger.log("creating model and diffusion...")
    model, diffusion = create_regressorLinear_and_diffusion(
        **args_to_dict(args, regressor_and_diffusion_defaults().keys())

    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())



    logger.log("creating data loader...")

    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     deterministic = False
    # )

    RegStruc = np.load('Output_12480.npy')
    RegHomo = np.load('Input_12480.npy')


    # RegStruc = RegStruc[:1000,:]
    # RegHomo = RegHomo[:, 0]

    # 计算最小值和最大值
    min_val = np.min(RegHomo)
    max_val = np.max(RegHomo)

    # # 执行归一化
    # Homo_scaled = (RegHomo - min_val) / (max_val - min_val)
    # Homo_scaled = RegHomo

    # 标准化变换
    scaler_Homo = StandardScaler()
    Homo_scaled = scaler_Homo.fit_transform(RegHomo)

    reversed_scale = scaler_Homo.inverse_transform(Homo_scaled)

    Homo_scaled = RegHomo[:,1:3]

    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            deterministic = False,
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training regressor model...")

    global batch_read_num
    batch_read_num = 0
    def forward_backward_log(Homo_scaled, prefix="train", print_res=False):
        global batch_read_num
        datasetsize = np.size(Homo_scaled, 0)

        # batch, batch_cons, _ = next(self.data)
        start_index = batch_read_num % datasetsize

        if datasetsize - start_index < 32:
            batch_struc = RegStruc[start_index:]

            batch_homo = Homo_scaled[start_index:]

            batch_read_num = 32 - (datasetsize - start_index)
        else:
            batch_struc = RegStruc[start_index:start_index + 32]
            batch_homo = Homo_scaled[start_index:start_index + 32]
            batch_read_num = (batch_read_num + 32) % datasetsize


        batch_struc = th.tensor(batch_struc).to(dtype=th.float32)


        # batch_homo = np.repeat(batch_homo, 11, axis=1)
        # batch_homo = batch_homo.reshape(32, 3, 11)
        batch_homo = th.tensor(batch_homo).to(dtype=th.float32)





        batch_struc = batch_struc.to(dist_util.dev())
        batch_homo = batch_homo.to(dist_util.dev())
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch_struc.shape[0], dist_util.dev())
            batch_struc = diffusion.q_sample(batch_struc, t)
        else:
            t = th.zeros(batch_struc.shape[0], dtype=th.long, device=dist_util.dev())
        # logits = model(batch_struc, timesteps=sub_t).reshape(sub_deflect.shape)
        logits = model(batch_struc, timesteps=t).reshape(batch_homo.shape)
        if print_res:
            print(logits, batch_homo)

        loss = F.mse_loss(logits, batch_homo)

        predicted_scaled_np = logits.detach().cpu().numpy()
        # reversed_pred = scaler_Homo.inverse_transform(predicted_scaled_np)
        train_scaled_np = batch_homo.detach().cpu().numpy()

        opt.zero_grad()
        loss.backward()

        opt.step()


        return loss


    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        loss = forward_backward_log(Homo_scaled,print_res=False)

        if step % 1000 == 0:
            print(loss)



    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=6e-4,
        weight_decay=0.2,
        anneal_lr=False,
        batch_size=8,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
        in_channels=1+3,
    )
    defaults.update(regressor_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
