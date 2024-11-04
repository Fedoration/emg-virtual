import datetime
import os
# import random
import sys
import configparser

from pathlib import Path

import numpy as np
import pylsl

from pylsl import StreamInlet, resolve_streams

# parsing init params
config = configparser.ConfigParser()
conf_path = 'D:/repos/SALUT_ML/conf.ini'
config.read(conf_path)
# for item in config.items():
#     print(f'Next:       {item}')
PATH_TO_SALUT_ML_DIR = Path(config['global']['salut_ml_dir'])
sys.path.insert(1, str(PATH_TO_SALUT_ML_DIR))


def get_streams(max_buflen: int):
    """
        max_buflen (int): lsl buffer size in seconds
    """
    # TODO why resolve all?
    streams = resolve_streams()
    print(streams)
    for stream in streams:
        print('STREAM NAME', stream.name())
        if stream.name() == 'hand':
            inlet_vr = StreamInlet(stream, max_buflen=max_buflen, processing_flags=pylsl.proc_ALL)
        elif stream.name() == 'NVX24_Data':
            inlet_myo = StreamInlet(stream, max_buflen=max_buflen, processing_flags=pylsl.proc_ALL)
        elif stream.name() == 'band':
            inlet_myo = StreamInlet(stream, max_buflen=max_buflen, processing_flags=pylsl.proc_ALL)

    return inlet_vr, inlet_myo


def get_save_path(save_folder: os.PathLike, saved_samples: int):
    # name = f"{datetime.datetime.utcnow().isoformat()}__{saved_samples}.npz"
    # name = f'{random.randint(1, 100)}'
    name = f"{datetime.datetime.utcnow().strftime('%H_%M_%S_%f__%m_%d_%Y')}__{saved_samples}.npz"
    print('Name save file', name)
    save_path = Path(save_folder) / name
    print('Path save file', save_path)
    print('Path save file', save_path.parent.exists())

    return str(save_path)


def lsl_data_collection(channel_number_vr,
                        channel_number_myo,
                        lsl_max_buflen,
                        items_in_sample,
                        save_folder):

    inlet_vr, inlet_myo = get_streams(max_buflen=lsl_max_buflen)

    print('Recording in progress')

    # init before cycle
    sample_data_vr = np.full((items_in_sample, channel_number_vr), fill_value=np.nan)
    sample_timestamps_vr = np.full(items_in_sample, fill_value=np.nan)

    sample_data_myo = np.full((items_in_sample, channel_number_myo), fill_value=np.nan)
    sample_timestamps_myo = np.full(items_in_sample, fill_value=np.nan)

    counter = 0
    saved_samples = 0

    try:
        while True:
            sample_vr, timestamp_vr = inlet_vr.pull_sample(timeout=0.0)
            print('Pulled vr sample')
            #print(type(sample_vr))
            #print(sample_vr)

            sample_myo, timestamp_myo = inlet_myo.pull_sample()
            print('Pulled myo sample')
            # TODO are you sure if timestamps is something appropriate
            sample_timestamps_myo[counter] = timestamp_myo
            sample_data_myo[counter, :] = sample_myo

            if sample_vr is not None:
                sample_vr = np.array(sample_vr)
                sample_timestamps_vr[counter] = timestamp_vr
                print(f'Sample_vr shape is:  {sample_vr.shape}')
                sample_data_vr[counter, :] = sample_vr

                if counter % items_in_sample == 0:
                    print(sample_vr[1:5])
            else:
                print('Sample_vr is None')

            counter += 1
            if counter == items_in_sample:
                # Save to new npz
                save_path = get_save_path(save_folder, saved_samples)
                sample_data_vr_resh = sample_data_vr.reshape(sample_data_vr.shape[0], 16, -1)


                print(f'\n\nSave path:  {save_path}\n\n')
                print()
                np.savez(save_path,
                         data_myo=sample_data_myo, myo_ts=sample_timestamps_myo,
                         data_vr=sample_data_vr_resh, vr_ts=sample_timestamps_vr)

                print(sample_data_myo[:3], sample_data_vr_resh[:3])

                print(f'Saved {saved_samples} sample to {save_path}')

                saved_samples += 1
                counter = 0

                sample_data_vr[:] = np.nan
                sample_timestamps_vr[:] = np.nan
                sample_data_myo[:] = np.nan
                sample_timestamps_myo[:] = np.nan

                continue

    except KeyboardInterrupt:
        inlet_vr.close_stream()
        inlet_myo.close_stream()


if __name__ == '__main__':
    channel_number_vr = 240  # w/ realhand data collection and w/o: channel_number_vr = 128
    channel_number_myo = 8
    lsl_max_buflen = 1
    SR = 250  # myo fps
    # seconds_in_sample = 4
    seconds_in_sample = float(config['realtime_training']['continuous_seconds_in_sample'])
    items_in_sample = int(SR * seconds_in_sample)  # TODO test latency by seconds_in_sample
    save_folder = 'D:/repos/SALUT_ML/realtime_training/realtime_data'

    lsl_data_collection(channel_number_vr,
                        channel_number_myo,
                        lsl_max_buflen,
                        items_in_sample,
                        save_folder)
