import time
import math #, sys
import joblib
import numpy as np
import warnings
import threading

from pylsl import StreamInlet, resolve_streams

#from filters import ButterFilterRealtime, NotchFilterRealtime


def preprocess_sample(sample,std):
    sample_norm = sample/std[:,:6]
    return sample_norm

'''

class Preprocessor:
    def __init__(self, freq, notch_freqs, srate, btype, order=4):
        print(freq, notch_freqs, srate)
        self.filter_butter = ButterFilterRealtime(freq, srate, btype, order)
        #self.filter_notch = NotchFilterRealtime(notch_freqs, Q=5, fs=srate)

    def __call__(self, sample):
        sample = self.filter_butter(sample)
        #sample = self.filter_notch(sample)
        return sample

'''

def stream_processing(queue_output=None, stop_event=None):
    #print('Using default parameters for inference')
    stream_name = 'NVX24_Data'

    #freq = 30
    #notch_freqs = 50
    window_size = 250 # for 250 Hz

    model = joblib.load('../real_time_regression/Fedor_TEST.pickle')
    std =np.load('../real_time_regression/preproc_params_Fedor_TEST.npy')

    streams = resolve_streams()
    for stream in streams:
        print(stream.name())

        if stream.name() == stream_name:
            inlet_mio = StreamInlet(stream, max_buflen=1) # ????
            n_channels = 6#stream.channel_count()
            srate = stream.nominal_srate()
            print('Number of channels:', n_channels)

    #preprocessor = Preprocessor(freq, notch_freqs, srate, btype='highpass')
    sample_window = np.zeros((n_channels, window_size))
    batch_size = round(srate * 0.05) # 20 ms
    print(sample_window.shape, batch_size)

    sample_batch = []

    counter = 0
    time_pass_start = time.perf_counter()
    time_model_total = 0
    time_filter_total = 0
    time_model_start = 0
    time_model_stop = 0
    
    while not stop_event.is_set():
        sample, timestamp = inlet_mio.pull_sample()
        if sample is None:
            continue

        sample_batch.append(sample)
        
        if len(sample_batch) < batch_size:
            continue

        sample_batch_array = np.stack(sample_batch).T
        
        time_filter_start = time.perf_counter()
        #sample_batch_array = preprocessor(sample_batch_array)
        time_filter_stop = time.perf_counter()
        time_filter_total += (time_filter_stop - time_filter_start)
        
        # sample_batch_np: C x T
        sample_batch = []

        sample_window = np.concatenate([sample_window[:,batch_size:], sample_batch_array[:6]], axis=-1)

        features = preprocess_sample(sample_window[None,:,:],std[:,:,None])
        
        #print(features)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                time_model_start = time.perf_counter()
                
                
                
                
                
                prediction = model.predict(features)
                
                #print('prediction: ',prediction)
                #prediction = prediction[0]
                
                time_model_stop = time.perf_counter()
                time_model_total += (time_model_stop - time_model_start)
                
                #prediction_best = np.argmax(prediction)
                #prediction_proba = prediction[prediction_best]
                #if prediction_proba > 0.70:
                queue_output.put(prediction[0,:].tolist())#.item())
                
            except RuntimeWarning:
                print("Caught RuntimeWarning: BAD MATRIX!!!")
            
        
        counter += 1
        if counter % 10 == 0:
            time_pass_stop = time.perf_counter()
            # print(
            #     f'pass: {time_pass_stop - time_pass_start:.3f}s '
            #     f'filter: {time_filter_stop - time_filter_start:.3f}s '
            #     f'filter_total: {time_filter_total:.3f}s '
            #     f'model: {time_model_stop - time_model_start:.3f}s '
            #     f'model_total: {time_model_total:.3f}s '
            #     )
            print()
            time_pass_start = time_pass_stop
            time_filter_total = 0
            time_model_total = 0
            




if __name__ == '__main__':
    import queue
    q = queue.Queue()
    stop_event = threading.Event()
    stream_processing(q, stop_event)