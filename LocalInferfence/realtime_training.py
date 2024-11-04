import sys
import os
from pathlib import Path
from functools import partial
from collections import OrderedDict
from  threading import Thread
from typing import TypeVar
PathLike = TypeVar("PathLike", str, Path)
import time
import configparser


# Extern libs (pip install)
import pylsl
import torch
import numpy as np
from einops import rearrange
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# parsing init params
config = configparser.ConfigParser()
conf_path = 'D:\\repos\\SALUT_ML\\inference_vis\\actual_inference_scripts\\conf.ini'
config.read(conf_path)
PATH_TO_SALUT_ML_DIR = Path(config['global']['salut_ml_dir'])
sys.path.insert(1, str(PATH_TO_SALUT_ML_DIR))

from utils import data_utils, losses
from utils.hand_visualize import Hand, save_animation_mp4, visualize_and_save_anim, merge_two_videos, merge_two_gifs, visualize_and_save_anim_gifs
from utils.inference_utils import make_inference, calculcate_latency, get_angle_degree


CKPT_PATH = (PATH_TO_SALUT_ML_DIR 
             / Path(config['inference']['init_weights_path']))
REALTIME_WEIGHTS_FOLDER = (Path(config['realtime_training']['work_path']) 
                        / Path(config['realtime_training']['weights_folder'])) 
# temporary not from config
# REALTIME_DATASET_FOLDER = Path('D:\\SALUT_start\\realtime_training_build\\output')  

REALTIME_DATASET_FOLDER = (Path(config['realtime_training']['work_path']) 
                        / Path(config['realtime_training']['data_folder'])) 

DEVICE = config['realtime_training']['device']
REORDER_ELECTORDES = int(config['realtime_training']['reorder_electrodes'])
MODEL_TYPE = config['global']['model_type']
FREEZE_ENCODER = int(config['global']['freeze_encoder'])

def count_parameters(model): 
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    # print in millions
    print(f"Total: {n_total/1e6:.4f}M, Trainable: {n_trainable/1e6:.4f}M")

    return n_total, n_trainable

def load_HVATNetv3(CKPT_PATH = CKPT_PATH):
    from models import HVATNet_v3_FineTune

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hvatnet_v3_params =dict(n_electrodes=8, n_channels_out=20,
                        n_res_blocks=3, n_blocks_per_layer=3,
                        n_filters=128, kernel_size=3,
                        strides=(2, 2, 2), dilation=2,
                        use_angles=True)
    model = HVATNet_v3_FineTune.HVATNetv3(**hvatnet_v3_params)
    # Load weights
    model.load_state_dict(torch.load(CKPT_PATH, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.train()
    count_parameters(model)
    return model


class TrainConfig:
    
    init_weights = CKPT_PATH.name
    data_folder = REALTIME_DATASET_FOLDER
    path_to_weights = REALTIME_WEIGHTS_FOLDER
    vr_output_fps = 25
    myo_input_fps = 200
    input_window_size = 256
    epoch_time = float(config['realtime_training']['epoch_time'])
    samples_per_epoch = int(config['realtime_training']['samples_per_epoch'])
    petyaslava_p_out = int(config['realtime_training']['petyaslava_p_out'])
    # recalculate_sampling_distr = int(config['realtime_training']['recalculate_sampling_distr'])
    is_real_hand = False
    random_sampling = config['realtime_training']['random_sampling']
    myo_transform = None
    use_angles = True # use angels as target.

    max_epochs = int(config['realtime_training']['max_epochs'])
    train_bs = int(config['realtime_training']['train_bs'])
    val_bs = int(config['realtime_training']['val_bs'])
    device = DEVICE
    optimizer_params = dict(lr=float(config['realtime_training']['optimizer_lr']),
                            wd=float(config['realtime_training']['optimizer_wd'])) 
    

    def _get_attrs(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
def train_loop(model, config):
    # model = model.float()
    dataset = data_utils.VRHandMYODatasetRealtime(
            data_folder = config.data_folder,
            vr_output_fps = config.vr_output_fps,
            input_window_size = config.input_window_size,
            samples_per_epoch = config.samples_per_epoch, 
            petyaslava_p_out = config.petyaslava_p_out,
            myo_input_fps = config.myo_input_fps,
            is_real_hand = config.is_real_hand,
            random_sampling = config.random_sampling,
            myo_transform = config.myo_transform,
            use_angles = config.use_angles,
            debug_indexes=True
            )


    dataset.append_new_data()
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=config.train_bs,
                                            shuffle=False,
                                            num_workers=0,
                                            )
    
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config.optimizer_params['lr'], weight_decay=0)
    min_angle_rad = 100
    num_steps = 0
    

    for epoch in range(config.max_epochs):
        
        start_epoch_time = time.time()
        print(f'STARTED EPOCH {epoch}')
        running_loss = 0.0
        angle_rad = 0
        # len of dataloader should be big
        for i, data in enumerate(dataloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            if dataloader.dataset.debug_indexes:
                inputs, labels, output_indexes, input_indexes, = data
                # continue
            else:
                inputs, labels = data
            
            if REORDER_ELECTORDES:
                inputs = inputs[:, (6, 5, 4, 3, 2, 1, 0, 7), :]

            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            # zero the parameter gradients
            optimizer.zero_grad()


            if MODEL_TYPE == 'hvatnet_v3':            
                # default hvatnet step.
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                angle_rad += loss.item()
            else:
                raise ValueError('hvatnet_v2 is not supported')    
            
            num_steps += 1
            if time.time() - start_epoch_time > config.epoch_time:
                break

        end_epoch_time = time.time()

        
         
        with torch.no_grad():
            avg_angle_rad = angle_rad/(i+1)
            avg_angle_degree = avg_angle_rad*180 / np.pi

            start_appending_time = time.time()
            prev_vr_len = len(dataset)
            dataset.append_new_data()
            end_appending_time = time.time()


            # start_val_time = time.time()

            # if prev_vr_len != len(dataset):
            #     print('!!!!!!!!!!!!! VALIDATION !!!!!!!!!!!!!!!!!')
            #     val_myo, val_vr = dataset.get_slice_from_outind_to_end(prev_vr_len)
            #     print(f'{val_myo.shape}, {val_vr.shape}, {len(val_myo) / len(val_vr) = }')

            #     # eval model on newly appended data
            #     if REORDER_ELECTORDES:
            #         val_myo = val_myo[:, (6, 5, 4, 3, 2, 1, 0, 7)]

            #     val_myo = torch.Tensor(val_myo).to(config.device)
            #     val_vr = torch.Tensor(val_vr).to(config.device)

            #     val_vr = rearrange(val_vr, '(b t) c -> b c t',
            #             b = int(len(val_vr) / config.input_window_size * 8),
            #             t=config.input_window_size // 8)
            #     # transformer step.
            #     if MODEL_TYPE == 'handformer':
            #         val_myo = rearrange(val_myo,
            #                             '(b t) c -> b 1 c t',
            #                             b = int(len(val_myo) / config.input_window_size),
            #                                 t = config.input_window_size)

            #         if DEVICE == 'cuda':
            #             with torch.cuda.amp.autocast():
            #                 val_loss, val_pred = model(val_myo, targets=val_vr, train=False)  
            #         else:    
            #             val_loss, val_pred = model(val_myo, targets=val_vr, train=False)
            #         val_pred = rearrange(val_pred, 'b t c -> (b t) c')    
                        
            #     elif MODEL_TYPE == 'hvatnet_v3':            
            #         # default hvatnet step.
            #         val_myo = rearrange(val_myo,
            #                             '(b t) c -> b c t',
            #                             b = int(len(val_myo) / config.input_window_size),
            #                                 t = config.input_window_size)
            #         val_pred = model(val_myo)
            #         val_pred = rearrange(val_pred, 'b c t -> (b t) c')

            #     val_vr = rearrange(val_vr, 'b c t -> (b t) c')
            #     # calculate metrics
            #     print('! VAL SHAPES ------------------')
            #     print('pred', val_pred.shape)
            #     print('myo', val_myo.shape)
            #     print('vr', val_vr.shape)
            #     dif = torch.mean(torch.abs(val_vr - val_pred)).cpu().numpy()
            #     angle_degree = np.round(np.rad2deg(dif), 3)

            #     corr_coef = torch.mean(torch.nn.functional.cosine_similarity(val_vr,
            #                                                                 val_pred,
            #                                                                 dim=0,
            #                                                                 eps=1e-8))
            #     corr_coef = np.round(corr_coef.item(), 3)
            # else:
            #     print('no new data received -> no validation')    
            # end_val_time = time.time()

        

        start_model_saving_time = time.time()
        save_path = REALTIME_WEIGHTS_FOLDER / f'{MODEL_TYPE}_{str(num_steps).zfill(5)}_{np.round(avg_angle_degree, 2)}.pt'
        torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
        end_model_saving_time = time.time()
        # latency
        epoch_time = end_epoch_time - start_epoch_time
        appending_time = end_appending_time - start_appending_time
        model_saving_time = end_model_saving_time - start_model_saving_time
        print(f""" Latency research {epoch=}:
        {avg_angle_degree=},
        {epoch_time=},
        {appending_time=},
        {model_saving_time=}
        """)    
  
if __name__ == '__main__':
    print(TrainConfig._get_attrs(TrainConfig))
    if MODEL_TYPE == 'hvatnet_v3':
        load_model = load_HVATNetv3
    else:
        raise ValueError('Wrong model_type or weights ckpt/pt')     
    print()
    model = load_model(CKPT_PATH = CKPT_PATH)
    train_loop(model, config=TrainConfig)


