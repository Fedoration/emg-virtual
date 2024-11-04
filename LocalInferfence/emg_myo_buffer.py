from raw_200Hz import Myo, create_lsl_outlet, emg_mode

if __name__ == '__main__':
    outlet, srate = create_lsl_outlet(srate=200, name='band',
                                        type='EMG', n_channels=8,
                                        dtype = 'int8', uid = 'uid1488')
    outlet_quat, srate_quat = create_lsl_outlet(srate=50, name='physics',
                                        type='quat', n_channels=4,
                                        dtype = 'float32', uid = 'uid10')
    # srate, outlet = lsl_info_get(srate=200, name='predict', type='points', n_channels=64)
    # srate_quat, outlet_quat = lsl_info_get(srate=50, name='physics', type='quat', n_channels=4)

    def proc_emg(emg, moving):
        print(emg)
        outlet.push_sample(emg)
        
    def proc_quat(quat, gyro, acc):
        outlet_quat.push_sample(quat)

    m=Myo(mode=emg_mode.RAW)
    m.add_emg_handler(proc_emg)
    # m.sleep_mode(1)
    m.add_imu_handler(proc_quat)
    m.connect()
    m.vibrate(1)

    try:
        while True:
            m.run()

    except KeyboardInterrupt:
        m.disconnect()
        quit()