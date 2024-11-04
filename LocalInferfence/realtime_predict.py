import sys
import os
from pathlib import Path
from functools import partial
import configparser

from typing import TypeVar
import time
# Extern libs (pip install)
import pylsl
import torch
import numpy as np
from einops import rearrange
import pytorch_lightning as pl


# parsing init params
config = configparser.ConfigParser()
conf_path = 'D:\\repos\\SALUT_ML\\inference_vis\\actual_inference_scripts\\conf.ini'
config.read(conf_path)
PATH_TO_SALUT_ML_DIR = Path(config['global']['salut_ml_dir'])
sys.path.insert(1, str(PATH_TO_SALUT_ML_DIR))

# local modules
from utils.quats_and_angles import get_quats
from raw_200Hz import create_lsl_outlet

CKPT_PATH = (PATH_TO_SALUT_ML_DIR 
             / Path(config['inference']['init_weights_path']))
REALTIME_WEIGHTS_FOLDER = (Path(config['realtime_training']['work_path']) 
                        / Path(config['realtime_training']['weights_folder']))   
DEVICE = config['inference']['device']
# DEVICE = 'cuda'
REORDER_ELECTORDES = int(config['realtime_training']['reorder_electrodes'])
MODEL_TYPE = config['global']['model_type']

exec_time_list = []

PathLike = TypeVar("PathLike", str, Path)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

def normalize_quats(v):
    """
    [Time, n_bones, 4]
    """
    norm = np.linalg.norm(v, axis = -1, keepdims=True)
    return v / norm

def smooth_ema(data, coef, prev=None):
    """
    This should not be applied on quanternions
    [Time, ...]
    coef in range [0, 1)
    coef = 0 -> no smooth
    """
    
    if prev is None:
        prev = data[0]
        for i in range(1, len(data)): 
            data[i] = prev * coef + data[i] * (1 - coef)
            prev = data[i]
    else: 
        for i in range(0, len(data)): 
            data[i] = prev * coef + data[i] * (1 - coef)
            prev = data[i]        
    return data

# method for quats model in same fps as EMG
def myo_to_vr(emg, device=DEVICE):
    """
    Preproc emg window and predict vr points.
    Last points only.

    :return:
        Return list of  Last vr points with step parameters [step, 16*4]
    """
    # global targets_load
    try:
        emg = (np.array(emg) + np.nanmean(np.array(emg))) / np.nanmax(np.array(emg))
        # CHANGES
        # emg = (np.array(emg) + 128) / 255.
    except Exception as err:
        raise err
    
    vr_output = model.inference(emg, device=device, first_bone_is_constant=True)
    vr_output = np.reshape(vr_output, (vr_output.shape[0], 64))
    
    # vr_last_points = vr_output[-STRIDE:]  # get last points but with step border
    vr_last_points = vr_output[::MODEL_DS_RATE]  # downsample
    vr_last_points = normalize_quats(vr_last_points)

    return list(vr_last_points)

# for angle model use inference_v2
def myo_to_angles(emg, device=DEVICE):
    """
    Preproc emg window and predict vr points.
    Last points only.

    :return:
        Return list of  Last vr points with step parameters [step, 16*4]
    """
    # global targets_load
    try:
        emg = (np.array(emg) + np.nanmean(np.array(emg))) / np.nanmax(np.array(emg))
        # CHANGES
        # emg = (np.array(emg) + 128) / 255.
    except Exception as err:
        print(emg)
        raise err

    emg = 2 * emg - 1

    if REORDER_ELECTORDES:
        emg = emg[:, (6, 5, 4, 3, 2, 1, 0, 7)]
    
        
    if MODEL_TYPE == 'hvatnet_v3':
        # for hvatnetmodel!!
        vr_output = model.inference_v2(emg, device=device, first_bone_is_constant=True)
    else: 
        raise ValueError('Wrong model_type !')
    return vr_output

## need to import HVATNet_v2 for this to work
def load_HVATNetv2(CKPT_PATH = 'HVATNet_v2_matvey_fixed_quats.pt'):
    from models import HVATNet_v2
     # init model and load weights
    hvatnet_v2_params =dict(n_electrodes=8, n_channels_out=64,
                        n_res_blocks=3, n_blocks_per_layer=2,
                        n_filters=128, kernel_size=3,
                        strides=(2, 2, 2, 4),
                        dilation=2)
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HVATNet_v2.HVATNetv2(**hvatnet_v2_params)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval()
    return model

def load_HVATNetv3(CKPT_PATH = Path(os.path.realpath(os.path.pardir)) /'ALVI_labs'/ 'SALUT_ML' /'weights'/'hvatnet_v3_angle.pt'):
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
    model.eval()
    return model


# buffer of EMG inlet shold be >= WINDOW_SIZE (maybe in seconds)
def pull_predict(emg_inlet,
                 quat_buf_outlet,
                    window_size,
                    stride,
                        emg_buffer,
                            last_angle=None,
                            smooth_coef = None):
    counter = 0 
    start_time = pylsl.local_clock()
    
    # dropping stride of old EMG
    emg_buffer = emg_buffer[stride:]
    # pulling stride new emg
    emgpull_start_time = pylsl.local_clock()
    while counter < stride:
        emg, timestamp = emg_inlet.pull_sample()
        emg_buffer.append(emg)
        counter += 1
    emgpull_end_time = pylsl.local_clock()

    # angle_buffer = myo_to_vr(emg_buffer)
    modelpred_start_time = pylsl.local_clock()
    angle_buffer = myo_to_angles(emg_buffer)

    modelpred_end_time = pylsl.local_clock()

    angle_buffer = angle_buffer[-(stride//MODEL_DS_RATE):]

    # smoothing only to angles
    smooth_start_time = pylsl.local_clock()
    if not smooth_coef is None:
        angle_buffer = smooth_ema(data=angle_buffer,
                                    prev=last_angle,
                                    coef=smooth_coef)
    smooth_end_time = pylsl.local_clock()    
    # angles to quats conversion
    anglequat_start_time = pylsl.local_clock()
    quat_to_push = get_quats(angle_buffer)
    quat_to_push = rearrange(quat_to_push, 't b q -> t (b q)')
    anglequat_end_time = pylsl.local_clock()
    # this be buffered again and resend with regular sample rate
    for sample in quat_to_push:
        quat_buf_outlet.push_sample(sample)
        
    # counting exec time and wait for new stride of EMG samples to come 
    end_time = pylsl.local_clock()
    exec_time = end_time  - start_time   
    wait_time = stride/VR_OUTPUT_FPS - exec_time # waiting for stride of EMG to come
    exec_time_list.append(float(exec_time))
    # logging execution time
    time_log_pull.append(emgpull_end_time - emgpull_start_time) 
    time_log_modelpred.append(modelpred_end_time - modelpred_start_time) 
    time_log_smooth.append(smooth_end_time - smooth_start_time)
    time_log_exec.append(end_time - start_time)
    print(f'''
    emg pull time = {time_log_pull[-1]}
    model pred time = {time_log_modelpred[-1]}
    smoothing time = {time_log_smooth[-1]}
    EXEC time = {time_log_exec[-1]}
    ''')
    # time.sleep(0 if wait_time < 0 else wait_time)
    # return emg_buffer and last angle for smoothing
    return emg_buffer, angle_buffer[-1]

def event_loop(model, 
               emg_inlet,
               quat_buf_outlet,
               window_size,
               stride,
               smooth_coef=None, 
               update_weights_time = 15):
    emg_buffer = [[0 for j in range(8)] for i in range(window_size)]
    prev_angle = None
    print(len(emg_buffer))
    counter = 0
    global_counter = 0
    last_weights_path = None
    while True: 
        emg_buffer, prev_angle = pull_predict(emg_inlet=emg_inlet,
                                              quat_buf_outlet=quat_buf_outlet,
                                                window_size=window_size,
                                                stride=stride,
                                                emg_buffer=emg_buffer,
                                                last_angle=prev_angle,
                                                smooth_coef=smooth_coef)
        if stride / VR_OUTPUT_FPS * counter > update_weights_time:
            counter = 0 
            start_update_time = time.time()
            model, last_weights_path, updated =  reload_weights(model, REALTIME_WEIGHTS_FOLDER, last_weights_path)
            update_time = time.time() - start_update_time
            print(f'weights {updated=} {last_weights_path.name if last_weights_path is not None else last_weights_path}')
            if updated:
                print(f'WEIGHTS UPDATE TIME = {update_time}')
        counter+=1
        global_counter += 1
    
def reload_weights(model, folder, last_weights_path):
    paths = sorted(folder.iterdir())
    if len(paths) > 2:
        new_path = sorted(folder.iterdir())[-2]
    else:
        new_path = last_weights_path
    # print(sorted(folder.iterdir()))
    print(f'{new_path=}')
    print(f'{last_weights_path=}')
    updated = (last_weights_path != new_path) 
    if updated:
        model.load_state_dict(torch.load(new_path, map_location=torch.device(DEVICE)))
        model.to(DEVICE)
        model.eval()
    return model, new_path, updated     

if __name__ == '__main__':
    # parse config
    MYO_INPUT_FPS = int(config['global']['myo_input_fps'])
    VR_OUTPUT_FPS = int(config['global']['vr_output_fps']) # fps which we want to have in vr. 
    MODEL_OUTPUT_FPS = int(config['global']['model_output_fps'])
    WINDOW_SIZE = int(config['inference']['window_size'])
    STRIDE = int(config['inference']['stride'])
    SMOOTH_COEF = float(config['inference']['smooth_coef'])
    UPDATE_WEIGHTS_TIME = int(config['inference']['update_weights_time'])

    TOTAL_DS_RATE = MYO_INPUT_FPS // VR_OUTPUT_FPS
    MODEL_DS_RATE = MYO_INPUT_FPS // MODEL_OUTPUT_FPS
    
    VR_BUFFER = []
    EMG_BUFFER = []
    counter_emg = 0
    
    streams = pylsl.resolve_stream('name', 'NVX24_Data')
    #streams = pylsl.resolve_stream('type', 'EMG')
    conn_inlet = pylsl.StreamInlet(streams[0],
                                    max_buflen=2,
                                    processing_flags=pylsl.proc_ALL) # Buflen in seconds? 400 samples in buffer
                                                           # avg exec_time w/ all processing flags = 0.3203864203225768
                                                           # avg exec_time w/ no processing flags = 0.32050284853652466
    print('emg stream inlet resolved')
    outlet, srate = create_lsl_outlet(srate=250,
                                      name='predictIrregular',
                                      type='IrregularQuats',
                                      n_channels=64, 
                                      dtype = 'float32')

    model = load_HVATNetv3(CKPT_PATH)
    
    time_log_pull = [] 
    time_log_modelpred = [] 
    time_log_smooth = []
    time_log_exec = []
    try:
        event_loop(model = model,
                    emg_inlet = conn_inlet,
                     quat_buf_outlet=outlet,
                      window_size=WINDOW_SIZE,
                        stride=STRIDE, 
                          smooth_coef=SMOOTH_COEF,
                            update_weights_time = UPDATE_WEIGHTS_TIME)

    except KeyboardInterrupt:
        # m.disconnect()
        print(np.mean(exec_time_list))
        print(f'''!!! AVERAGE TIME !!!:
        emg pull time = {np.mean(time_log_pull)}
        model pred time = {np.mean(time_log_modelpred)}
        smoothing time = {np.mean(time_log_smooth)}
        EXEC time = {np.mean(time_log_exec)}
        ''')
        quit()
# cd inference_vis/actual_inference_scripts/
# pipenv run python realtime_predict.py 