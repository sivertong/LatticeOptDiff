"""
Train the main diffusion model (regardless of guidance) on images.
"""

import argparse

import dist_util, logger
from resample import create_named_schedule_sampler
from script_util_IH import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from train_util_IH import TrainLoop

import numpy as np

import joblib  # 导入joblib

import scipy.io as sio

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    # 现读取函数
    data_structure = np.load('Output_37437.npy')
    data_label1 = np.load('Input_37437.npy')# design variables

    # 读取归一化数据
    data_label = np.load('normalized_Input_37437.npy') # material properties

    # 从DropNum-matlab文件中读取训练集
    TrainData = sio.loadmat("D:\CodeSave\GitCode\IHDiff\\topodiff\MainDiff_IO_Data_DropNum.mat")
    data_label = TrainData['MainDiffHomo']
    data_structure = TrainData['MainDiffStruc']
    # 对数据进行归一化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_label = scaler.fit_transform(data_label)
    joblib.dump(scaler, 'scaler_36316.joblib')


    # data_structure = np.load('./ocean.npy')
    # data_label = np.load('./ocean.npy')
    data_structure = np.array(data_structure, dtype = np.float32)
    data_label = np.array(data_label, dtype = np.float32)



    '''
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond
    )
    '''


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data_structure=data_structure,
        data_label=data_label,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="", 
        schedule_sampler="loss-second-moment",
        lr=1e-4,
        weight_decay=0.1,
        lr_anneal_steps=0,
        batch_size=256,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
