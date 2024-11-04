import time
import pylsl
from raw_200Hz import create_lsl_outlet
from collections import deque
from  threading import Thread
outlet, srate = create_lsl_outlet(srate=25,
                    name='predict',
                    type='points',
                    n_channels=64,
                    dtype = 'float32',
                    uid = 'uid228')

    
streams = pylsl.resolve_stream('type', 'IrregularQuats')
print(streams)
# create a new inlet to read from the stream
conn_inlet = pylsl.StreamInlet(streams[0])

stride = 64
ds_rate = 8
q = deque([], maxlen = stride//ds_rate)

def read_lsl():
    while True:
        sample, timestamp = conn_inlet.pull_sample() # blocking pull
        print('PULL ', timestamp)
        q.append(sample)

def pull_from_buffer():
    try:
        sample = q.popleft()
    except IndexError:
        sample = None
    return sample    

# variant 2
def push_to_buffer(sample):
    q.append(sample)
    print('Pushed to deque')


def push_to_lsl(sample):
    # here we should do everything from tish script, but in the tread of realtimepredict
    pass

# cd SALUT_ML/inference_vis/actual_inference_scripts/
# pipenv run python
# cd inference_vis/actual_inference_scripts/
# pipenv run python
counter = 0
fps = 25
interval = 1/fps
pulled_sample = None


t = Thread(target=read_lsl, args=[], daemon=True)
t.start()

latest_sample = pull_from_buffer()
while latest_sample is None:
    time.sleep(0.001)
    latest_sample = pull_from_buffer()
first_sample_timestamp = time.perf_counter()

while True:
    current_timestamp = time.perf_counter()
    delay = counter*interval - (current_timestamp - first_sample_timestamp) 
    if delay > 0:
        print('wait')
        time.sleep(delay)
        if pulled_sample is None:
            pulled_sample = pull_from_buffer()
            if pulled_sample is not None:
                latest_sample = pulled_sample 
        
    outlet.push_sample(latest_sample)
    print('PUSH')
    # lsl should be with buffer = stride*fps//orig_fps for example 64*25//200 = 8
    pulled_sample = pull_from_buffer()
    if pulled_sample is not None:
        latest_sample = pulled_sample
    counter += 1

















# # fps_prediction = 200
# # fps_vr_unity = 25
# srate = 25

# """
# Here we regularize fps of received quats
# """
# start_time = pylsl.local_clock()
# sent_samples = 0
# ts_prev = pylsl.local_clock()
# local_quat_buff = []
# QUAT_BUFFER = []
# exec_time_prev = 0
# fps = 25
# q = queue.Queue(3)


# first_timestamp = None
# counter = 0
# while True:
#     start_time = pylsl.local_clock()

    
#     mysample, ts = conn_inlet.pull_sample()
#     pull_timestamp = time.perf_counter()
#     counter += 1
#     if first_timestamp is None:
#         first_timestamp = pull_timestamp
    
# im
#     outlet.push_sample(mysample)


#     flag = round(ts-ts_prev, 3) < 1/fps
#     print('time_between = ', round(ts-ts_prev, 4), ' needed_time = ', 1/fps, ' flag = ', flag)
#     ts_prev = ts
    
#     # print(1/25 - exec_time)
#     # exec_time = pylsl.local_clock() - start_time
#     exec_time = ts-ts_prev
#     exec_time += exec_time_prev
    
#     ideal_timestamp = first_timestamp + counter/fps 
#     delay = ideal_timestamp - pull_timestamp
#     # TODO if delay > 0:
#     # TODO if delay < 0:
    
#     if  exec_time > 1/fps:
#         time.sleep(0)
#         exec_time_prev =  exec_time - 1/fps
        

#     else:
#         time.sleep(1/fps - exec_time)
#         exec_time_prev = 0 
#     print('exec_time (push)', round(exec_time, 3))
# while True:

#     elapsed_time = pylsl.local_clock() - start_time
#     # print(elapsed_time)
#     # number of samples to 
#     required_samples = int(srate * elapsed_time) - sent_samples
#     for sample_ix in range(required_samples):
#         # make a new random n_channels sample; this is converted into a
#         # pylsl.vectorf (the data type that is expected by push_sample)
#         # global index:
#         # global_index = sent_samples + sample_ix
#         # print('Sent sample number: ', global_index)
#         mysample, timestamp = conn_inlet.pull_sample()
#         # print(f'Inlet timestamp {timestamp}', 'pylsl.local_time ', pylsl.local_clock())
#         # now send it
#         outlet.push_sample(mysample)
#         print('Pushtime : ', pylsl.local_clock())
#     sent_samples += required_samples
#     # now send it and wait for a bit before trying again.
#     # time.sleep(1/25)
#     time.sleep(0.001)


