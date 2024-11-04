from pylsl import StreamInlet, resolve_streams
import numpy as np
import time

sample_sec = 90  # сколько длится сбор данных
channel_number_vr = 240
channel_number_myo = 8
SR = 520


def lsl_data_collection(gest_type, folder):
    sample_data_vr = np.zeros((SR * sample_sec, channel_number_vr + 1))  # один дополнительный для времени
    sample_data_vr = sample_data_vr * np.nan

    sample_data_myo = np.zeros((SR * sample_sec, channel_number_myo + 1))
    sample_data_myo = sample_data_myo * np.nan

    streams = resolve_streams()
    for stream in streams:
        print(stream)
        if stream.name() == 'hand':
            inlet_vr = StreamInlet(stream, max_buflen=1)
        elif stream.name() == 'NVX24_Data':
            inlet_myo = StreamInlet(stream, max_buflen=1)
        elif stream.name() == 'band':
            inlet_myo = StreamInlet(stream, max_buflen=1)

    time_start = time.time()
    counter = 0

    print('Recording in progress')

    while time.time() - time_start < sample_sec:
        sample_vr, timestamp_vr = inlet_vr.pull_sample(timeout=0.0)
        print(sample_vr)
        sample_myo, timestamp_myo = None, None
        sample_myo, timestamp_myo = inlet_myo.pull_sample()

        sample_data_myo[counter, 1:] = sample_myo
        sample_data_myo[counter, 0] = timestamp_myo
        print(sample_myo)
        if sample_vr is not None:
            sample_data_vr[counter, 1:] = sample_vr
            sample_data_vr[counter, 0] = timestamp_vr
            # print(sample_vr[1:5])

        counter += 1

    print('1')
    inlet_vr.close_stream()
    inlet_myo.close_stream()
    print('2')
    myo_timestamps = sample_data_myo[:, 0]
    data_myo = sample_data_myo[:, 1:]
    print('3')
    vr_timestamps = sample_data_vr[:, 0]
    data_vr = sample_data_vr[:, 1:].reshape(sample_data_vr.shape[0], 16, -1)

    filepath = folder + '/' + str(gest_type) + '_data_test_' + str(time.time())[4:10]
    np.savez(filepath, data_myo=data_myo, myo_ts=myo_timestamps,
             data_vr=data_vr, vr_ts=vr_timestamps)
    print('Saved')

    return


if __name__ == '__main__':
    lsl_data_collection()
