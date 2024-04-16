"""
Like image_sample.py, but use a noisy image regressor to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import scipy.io as sio

from topodiff.cons_input_datasets import load_data
from topodiff import dist_util, logger
from topodiff.script_util_IH import (
    model_and_diffusion_defaults,
    regressor_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_regressor,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    create_regressorMuRho,
)

import matplotlib.pyplot as plt


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading regressor...")
    regressor = create_regressor(regressor_depth = 4, in_channels = 1, **args_to_dict(args, regressor_defaults().keys()))
    regressor.load_state_dict(
        dist_util.load_state_dict(args.regressor_path, map_location="cpu")
    )
    regressor.to(dist_util.dev())
    if args.regressor_use_fp16:
        regressor.convert_to_fp16()
    regressor.eval()

    logger.log("loading Regressor2...")
    regressorMuRho = create_regressorMuRho(regressor_depth = 4, in_channels = 1, **args_to_dict(args, regressor_defaults().keys()))
    regressorMuRho.load_state_dict(
        dist_util.load_state_dict(args.regressorMuRho_path, map_location="cpu")
    )
    regressorMuRho.to(dist_util.dev())
    if args.regressorMuRho_use_fp16:
        regressorMuRho.convert_to_fp16()
    regressorMuRho.eval()

    data = load_data(
        data_dir=args.constraints_path,
    )

    def cond_fn_1(x, cons, t):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)

            # 这里我们重复6次，然后截断到64
            x_repeated = x_in.repeat(1, 1, 6)[:, :, :64]  # 现在形状为(32, 1, 64)
            # 为了获得(32, 1, 64, 64)的形状，我们需要在末尾添加一个维度并重复
            x_repeated = x_repeated.unsqueeze(3).repeat(1, 1, 1, 64)  # flag

            logits = regressor(x_repeated, t)
            # 计算MSE
            E_target = cons[:, 0, 0]
            logits = logits.squeeze()
            E_MSE = F.mse_loss(logits, E_target)
            grad = th.autograd.grad(E_MSE, x_in, allow_unused = True)[0]
            return (-1) * grad[:,0,:].reshape((32,1,11)) * args.regressor_scale

    def cond_fn_2(x, cons, t):
        with th.enable_grad():

            x_in = x.detach().requires_grad_(True)

            # 这里我们重复6次，然后截断到64
            x_repeated = x_in.repeat(1, 1, 6)[:, :, :64]  # 现在形状为(32, 1, 64)
            # 为了获得(32, 1, 64, 64)的形状，我们需要在末尾添加一个维度并重复
            x_repeated = x_repeated.unsqueeze(3).repeat(1, 1, 1, 64)  # flag

            logits = regressorMuRho(x_repeated, t)
            MuRho_target = cons[:, 1:3, 0]

            MuRho_MSE = F.mse_loss(logits, MuRho_target)
            grad = th.autograd.grad(MuRho_MSE, x_in, allow_unused=True)[0]
            return (-1) * grad[:, 0, :].reshape((32, 1, 11)) * args.regressorMuRho_scale

    def model_fn(x, t):
        return model(x, t)


    logger.log("sampling...")
    all_images = []
    batch_read_num = 0
    endflag = 0
    while endflag==0:
        model_kwargs = {}
        # model_kwargs = None # 暂时设置为None，否则将会报错
        sample_fn = (
            diffusion.p_sample_loop_IH if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # TestInput = np.load('test.npy')

        optdata = sio.loadmat('C:\\Users\long\Documents\GitHub\IH-GAN_CMAME_2022\cmpdata_422iter.mat')
        xPhys = optdata['xPhys'].reshape(-1, 1)
        yPhys = optdata['yPhys'].reshape(-1, 1)
        zPhys = optdata['zPhys'].reshape(-1, 1)
        optdata = np.column_stack((xPhys, yPhys, zPhys))
        TestInput = optdata

        TestInput = np.load('Input_300.npy') # design variables

        datasetsize = np.size(TestInput, 0)

        start_index = batch_read_num % len(TestInput)
        if datasetsize - start_index < 32:

            TestBatch1 = TestInput[start_index:]
            TestBatch2 = TestInput[:32 - (datasetsize - start_index)]
            TestBatch = np.concatenate((TestBatch1, TestBatch2), axis=0)
            batch_read_num = 32 - (datasetsize - start_index)
            endflag = 1
        else:
            TestBatch = TestInput[start_index:start_index + 32]
            batch_read_num = (batch_read_num + 32) % len(TestInput)

        TestBatch = np.repeat(TestBatch, 11, axis=1)
        TestBatch = TestBatch.reshape(len(TestBatch), 3, 11)
        TestBatch = th.tensor(TestBatch).to(dtype=th.float32)

        TestBatch = TestBatch.cuda()




        sample = sample_fn(
            model_fn,
            (args.batch_size, 1, 11),
            TestBatch,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn_1 = cond_fn_1,
            cond_fn_2 = cond_fn_2,
            device=dist_util.dev(),
        )
        sample = sample.permute(0, 2, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=20,
        batch_size=1,
        use_ddim=False,
        model_path="",
        regressor_path="",
        fm_classifier_path="",
        regressor_scale=1.0,
        classifier_fm_scale=1.0,
        constraints_path="",
        classifier_use_fp16=True,
        vf_regressor_use_fp16=False,
        fm_classifier_use_fp16=False,
        regressorMuRho_path="",
        regressorMuRho_use_fp16=True,
        regressorMuRho_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(regressor_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    
    main()