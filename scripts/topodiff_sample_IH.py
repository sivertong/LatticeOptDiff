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

    """
    logger.log("loading regressor...")
    regressor = create_regressor(regressor_depth = 4, in_channels = 1+3+2+2, **args_to_dict(args, regressor_defaults().keys()))
    regressor.load_state_dict(
        dist_util.load_state_dict(args.regressor_path, map_location="cpu")
    )
    regressor.to(dist_util.dev())
    if args.regressor_use_fp16:
        regressor.convert_to_fp16()
    regressor.eval()

    logger.log("loading fm_classifier...")
    fm_classifier = create_classifier(in_channels = 1, **args_to_dict(args, classifier_defaults().keys()))
    fm_classifier.load_state_dict(
        dist_util.load_state_dict(args.fm_classifier_path, map_location="cpu")
    )
    fm_classifier.to(dist_util.dev())
    if args.fm_classifier_use_fp16:
        fm_classifier.convert_to_fp16()
    fm_classifier.eval()


    data = load_data(
        data_dir=args.constraints_path,
    )

    def cond_fn_1(x, t):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = regressor(x_in, t)
            grad = th.autograd.grad(logits.sum(), x_in, allow_unused = True)[0]
            return (-1) * grad[:,0,:,:].reshape((1,1,64,64)) * args.regressor_scale

    def cond_fn_2(x, t):
        with th.enable_grad():
            y = np.array([1], dtype = int)
            x_in = x.detach().requires_grad_(True)
            logits = fm_classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view()]
            grad = th.autograd.grad(selected.sum(), x_in)[0]
            return grad[:,0,:,:].reshape((1,1,64,64)) * args.classifier_fm_scale
                """

    def model_fn(x, t):
        return model(x, t)


    logger.log("sampling...")
    all_images = []
    batch_read_num = 0
    endflag = 0
    while endflag==0:
        model_kwargs = {}
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
            cond_fn_1=None,
            cond_fn_2=None,
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
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(regressor_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    
    main()