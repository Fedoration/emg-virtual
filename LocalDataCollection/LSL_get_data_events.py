from pylsl import StreamInlet, resolve_streams
import time
import LSL_get_data
import uuid
import os
from datetime import date


today_recording_folder = (str(date.today().month).zfill(2) + '_'
                          + str(date.today().day).zfill(2) + '_'
                          + str(date.today().year))
folder = 'recorded_data/' + today_recording_folder + '/' + str(uuid.uuid4())
if not os.path.exists('recorded_data'):
    os.mkdir('recorded_data')
if not os.path.exists('recorded_data/' + today_recording_folder):
    os.mkdir('recorded_data/' + today_recording_folder)
os.mkdir(folder)


def find_start():

    print('Search started')
    streams = resolve_streams()
    for stream in streams:
        if stream.name() == 'events':
            inlet_events = StreamInlet(stream, max_buflen=1)

    try:
        sample_event, timestamp_event = inlet_events.pull_sample()

    except UnboundLocalError:
        print('Can not find stream')
        time.sleep(5)
        find_start()

    if sample_event is not None:
        gest_type = sample_event[0]
        print('Sample is being recorded')
        print(gest_type)
        LSL_get_data.lsl_data_collection(gest_type, folder)

    inlet_events.close_stream()
    find_start()


def main():
    time.sleep(1)
    find_start()


if __name__ == '__main__':
    main()
