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


import dist_util, logger
from script_util_IH import (
    model_and_diffusion_defaults,
    regressor_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_regressor,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    create_regressorMuRho,
    create_regressor_and_diffusion,
    regressor_and_diffusion_defaults,
)
import random
import torch
import os
import matplotlib.pyplot as plt
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    # args = create_argparser().parse_args()
    # seed_everything(42)
    
    # dist_util.setup_dist()
    # logger.configure()



    # logger.log("loading regressor...")
    # regressor, diffusion2 = create_regressor_and_diffusion(in_channels = 1, regressor_depth=4,
    #     **args_to_dict(args, regressor_and_diffusion_defaults().keys())
    # )
    # print(regressor)
    # regressor.to(dist_util.dev())
    # regressor.load_state_dict(
    #     dist_util.load_state_dict('./regconvmodel020000.pt', map_location="cpu")
    # )

    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict("imgmodel180000.pt", map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # logger.log("loading regressor...")
    # regressor = create_regressor(regressor_depth = 4, in_channels = 1, **args_to_dict(args, regressor_defaults().keys()))
    # regressor.load_state_dict(
    #     dist_util.load_state_dict('./regconvmodel020000.pt',  map_location="cpu")
    # )
    # regressor.to(dist_util.dev())
    # if args.regressor_use_fp16:
    #     regressor.convert_to_fp16()
    # regressor.eval()

    

    # model.load_state_dict(
    #             dist_util.load_state_dict(
    #                 args.resume_checkpoint, map_location=dist_util.dev()
    #             )
    #         )


    # logger.log("loading Regressor2...")
    # regressorMuRho = create_regressorMuRho(regressor_depth = 4, in_channels = 1, **args_to_dict(args, regressor_defaults().keys()))
    # regressorMuRho.load_state_dict(
    #     dist_util.load_state_dict('model040000.pt', map_location="cpu")
    # )
    # regressorMuRho.to(dist_util.dev())
    # if args.regressorMuRho_use_fp16:
    #     regressorMuRho.convert_to_fp16()
    # regressorMuRho.eval()


    def cond_fn_1(x, cons, t):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            # print('x_in', x_in.shape)
            bs = 21
            # 这里我们重复6次，然后截断到64
            x_repeated = x_in.repeat(1, 1, 6)[:, :, :64]  # 现在形状为(32, 1, 64)
            # 为了获得(32, 1, 64, 64)的形状，我们需要在末尾添加一个维度并重复
            x_repeated = x_repeated.unsqueeze(3).repeat(1, 1, 1, 64)  # flag

            logits = regressor(x_repeated, t)
            # 计算MSE
            E_target = cons[:, 0, 0]
            # print('E_target', E_target.shape)
            logits = logits.squeeze()
            # print('logits', logits.shape)
            E_MSE = F.mse_loss(logits, E_target)
            grad = th.autograd.grad(E_MSE, x_in, allow_unused = True)[0]
            # print('grad', grad.shape)
            return (-1) * grad[:,0,:].reshape((bs,1,11)) * 1

    def cond_fn_2(x, cons, t):
        with th.enable_grad():
            bs = 21
            x_in = x.detach().requires_grad_(True)

            # 这里我们重复6次，然后截断到64
            x_repeated = x_in.repeat(1, 1, 6)[:, :, :64]  # 现在形状为(32, 1, 64)
            # 为了获得(32, 1, 64, 64)的形状，我们需要在末尾添加一个维度并重复
            x_repeated = x_repeated.unsqueeze(3).repeat(1, 1, 1, 64)  # flag

            logits = regressorMuRho(x_repeated, t)
            MuRho_target = cons[:, 1:3, 0]

            MuRho_MSE = F.mse_loss(logits, MuRho_target)
            grad = th.autograd.grad(MuRho_MSE, x_in, allow_unused=True)[0]
            return (-1) * grad[:, 0, :].reshape((bs, 1, 11)) * args.regressorMuRho_scale

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

        # optdata = sio.loadmat('C:\\Users\long\Documents\GitHub\IH-GAN_CMAME_2022\cmpdata_422iter.mat')
        # xPhys = optdata['xPhys'].reshape(-1, 1)
        # yPhys = optdata['yPhys'].reshape(-1, 1)
        # zPhys = optdata['zPhys'].reshape(-1, 1)
        # optdata = np.column_stack((xPhys, yPhys, zPhys))
        # TestInput = optdata

        # TestInput = np.load('Input_300.npy') # design variables
        # TestInput[:, 2] = TestInput[:, 2] - 0.1

        # 3万数据集中抽取
        optdata = sio.loadmat("D:\CodeSave\GitCode\IHDiff\scripts\TestSet600_03_DropNum.mat")
        TestInput = optdata['TestHomo']
        TestInput[:, 2] = TestInput[:, 2] - 0.05

        from sklearn.preprocessing import StandardScaler
        import joblib  # 导入joblib
        scaler = joblib.load('D:\CodeSave\GitCode\IHDiff\scripts\scaler_36316.joblib')
        TestInput = scaler.transform(TestInput)


        # TestInput = np.load('./ocean.npy')[:200]
        TestInput = np.array(TestInput, dtype = np.float32)
        datasetsize = np.size(TestInput, 0)

        start_index = batch_read_num % len(TestInput)
        print('sb',start_index)
        bs =21
        if datasetsize - start_index < bs:

            TestBatch1 = TestInput[start_index:]
            TestBatch2 = TestInput[:bs - (datasetsize - start_index)]
            TestBatch = np.concatenate((TestBatch1, TestBatch2), axis=0)
            batch_read_num = bs - (datasetsize - start_index)
            endflag = 1
        else:
            TestBatch = TestInput[start_index:start_index + bs]
            batch_read_num = (batch_read_num + bs) % len(TestInput)

        # TestBatch = TestBatch.reshape([-1  , 12]).reshape(-1,1,12)
        

        TestBatch = np.repeat(TestBatch, 11, axis=1)
        TestBatch = TestBatch.reshape(len(TestBatch), 3, 11)
        TestBatch = th.tensor(TestBatch).to(dtype=th.float32)

        TestBatch = TestBatch.cuda()

        




        sample = sample_fn(
            model_fn,
            (TestBatch.shape[0], 1, 11),
            TestBatch,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn_1 = None,
            cond_fn_2 = None,
            device=dist_util.dev(),
        )
        sample = sample.permute(0, 2, 1)
        sample = sample.contiguous()
        print(sample.shape)### 32 11 1
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        print(len(all_images))
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join('./', f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=300,
        batch_size=1,
        use_ddim=False,
        model_path="",
        regressor_path="",
        fm_classifier_path="",
        regressor_scale=1.0,
        classifier_fm_scale=1.0,
        constraints_path="",
        classifier_use_fp16=False,
        vf_regressor_use_fp16=False,
        fm_classifier_use_fp16=False,
        regressorMuRho_path="",
        regressorMuRho_use_fp16=False,
        regressorMuRho_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(regressor_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def create_argparser2():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=6e-5,
        weight_decay=0.2,
        anneal_lr=False,
        batch_size=256,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="./model050000.pt",
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