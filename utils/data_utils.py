# Built-in
import logging
import random
import copy
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from typing import Union, NoReturn, Sequence, Optional
from pathlib import Path

from natsort import natsorted
from einops import rearrange
import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from audiomentations.core.transforms_interface import BaseWaveformTransform

from .hand_visualize import inverse_rotations
from .quats_and_angles import get_angles


logger = logging.getLogger(__name__)


# VLAD: Added filters
def butter_highpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype="highpass")
    return b, a


def butter_highpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_highpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y


# TODO print -> logger
# All numeric constants must be with comments
def get_all_subjects_pathes(datasets):
    """
    Scan each dataset by .npz files.
    After that we get parent and build set top o that.
    So we get train and test pathes.
    """
    ALL_PATHS = []
    for d_path in datasets:
        data_dir = Path(d_path)
        file_names = list(data_dir.glob("**/*.npz"))
        paths = list(set([f.parent for f in file_names]))
        ALL_PATHS.extend(paths)
    ALL_PATHS = list(set(ALL_PATHS))
    return ALL_PATHS


def check_conditions(my_list, path):
    """
    Check whether path has folder with similar name as my list.
    So we can filter by train/test, left/right or together.
    """
    one_in_list = False
    for value in path.parts:
        if value in my_list:
            one_in_list = True
            break
    return one_in_list


def filter_by_condition(paths, condition):
    """
    Apply check condition for each path.
    Create new path list with "good" datasets.
    """
    FILTERED_PATHS = []
    for p in paths:
        if check_conditions(condition, p):
            FILTERED_PATHS.append(p)
    return FILTERED_PATHS


def get_train_val_pathes(config):
    """
    Config has to have ->
    config.datasets | config.human_type | config.hand_type | config.test_dataset_list
    Return:
    train and test pathes.
    """
    ALL_PATHS = get_all_subjects_pathes(config.datasets)

    FILTERED_PATHS = filter_by_condition(ALL_PATHS, config.human_type)
    FILTERED_PATHS = filter_by_condition(FILTERED_PATHS, config.hand_type)
    # FILTERED_PATHS = filter_by_condition(FILTERED_PATHS,)

    train_paths = filter_by_condition(FILTERED_PATHS, ["train"])
    test_paths = filter_by_condition(FILTERED_PATHS, ["test"])
    # print(test_paths)
    # test_paths = filter_by_condition(test_paths, config.test_dataset_list)

    return sorted(train_paths), sorted(test_paths)


def load_data_from_one_exp(file_path: Union[Path, str]) -> dict["str", np.ndarray]:
    # np.load loads data from *.npz lazy so filedescriptor must be closed
    with np.load(file_path) as file:
        exp_data = dict(file)
    return exp_data


def interpolate_quats_for_one_exp(
    data: dict[str, np.ndarray], quat_interpolate_method: str = "slerp"
) -> dict[str, np.ndarray]:
    """
    Inplace fill nan in quaternion_rotation positions (i.e. [:, :, 4:] slice) in data['data_vr']
    with interpolated quaternions based on existed values

    Args:
        data: dict with keys 'data_vr', 'data_myp', 'myo_ts', 'vr_ts' and corresonding np.ndarray values
        quat_interpolate_method (str): 'slerp' or 'nearest'(NotImplemented)

    Notes:
        (1) This function assume that vr_timestamps[0] and vr_timestamps[-1] is not np.nan

    Raises:
        ValueError: if myo_timestamps contains np.nan
    """

    #     data = copy.deepcopy(data)

    data_vr: np.ndarray = data["data_vr"]
    myo_timestamps: np.ndarray = data["myo_ts"]
    vr_timestamps: np.ndarray = data["vr_ts"]

    bones_amount = data_vr.shape[1]
    myo_timestamps = myo_timestamps[~np.isnan(myo_timestamps)]
    for_interp = len(myo_timestamps)
    if np.isnan(myo_timestamps).any():
        raise ValueError("myo_timestamps contains np.nan")

    # find mask for not nan positions
    vr_timestamps_mask = ~np.isnan(vr_timestamps)
    masked_data_vr = data_vr[vr_timestamps_mask]
    masked_vr_timestamps = vr_timestamps[vr_timestamps_mask]

    # We will get interpoletion function for quats with vr_timestamps, but we would like to get quats for timestamps
    # for each myo_timestamps what require them to be inside [vr_ts[0], vr_ts[-1]]
    # so all data will be sliced over time to satisfy this requirement
    new_left_idx = np.argmax(myo_timestamps >= masked_vr_timestamps[0])
    new_right_idx = myo_timestamps.shape[0] - np.argmax(
        np.flip([myo_timestamps <= masked_vr_timestamps[-1]])
    )
    print(f"Slice myo_timestamps and all data from {new_left_idx} to {new_right_idx}")

    if quat_interpolate_method == "slerp":
        # iterate over each bones and append results to list 'interpolated_quats'
        # which then will be stacked over axis=1 to np.ndarray object
        interpolated_quats = []
        for bone_idx in range(bones_amount):
            # TODO move one iteration to single function
            _bone_quats = masked_data_vr[:, bone_idx, 4:8]
            _rotations = Rotation.from_quat(_bone_quats)
            slerp = Slerp(masked_vr_timestamps, _rotations)
            myo_timestamps = np.linspace(
                masked_vr_timestamps[0], masked_vr_timestamps[-1], num=for_interp
            )
            _quats = slerp(myo_timestamps).as_quat()
            interpolated_quats.append(_quats)

        interpolated_quats = np.stack(interpolated_quats, axis=1)

    elif quat_interpolate_method == "nearest":
        raise NotImplementedError

    # data is already deepcopy of original data passed to function so we dont need to make another copy
    # and can overwrite current values inside data
    data["data_vr"][new_left_idx:new_right_idx, :, 4:8] = interpolated_quats
    sliced_data = {k: v[new_left_idx:new_right_idx] for k, v in data.items()}

    return sliced_data


def strip_nans_for_one_exp(
    data: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], int, int]:
    """
    Slice all array in dicts like arr[left_idx: right_idx + 1],
    where left_idx and right_idx are gotten from data['vr_ts']:

        [np.nan, np.nan, 2, 4, np.nan, 0.1, np.nan, 0.5, np.nan]
                         ^...........................^
                         |...........................|
                         .............................
                         left_idx....................right_idx

    """
    vr_timestamps: np.ndarray = data["vr_ts"]

    # find first and most latest positions with not np.nan values
    not_non_position = np.argwhere(~np.isnan(vr_timestamps))

    if len(not_non_position) == 0:
        return None, None, None

    left_idx = not_non_position[0][0]
    right_idx = not_non_position[-1][0]

    assert not np.isnan(vr_timestamps[left_idx])
    assert not np.isnan(vr_timestamps[right_idx])

    # arr[right_idx] is latest not nan item so slice must be before right_idx + 1 not inclusive
    stripped_data = {k: v[left_idx : right_idx + 1] for k, v in data.items()}
    return stripped_data, left_idx, right_idx


def calc_probas(iterable: Sequence[Sequence]) -> list:
    lens = []
    for child_iter in iterable:
        lens.append(len(child_iter))
    probas = np.array(lens) / sum(lens)
    return probas


def calc_stripped_len(sequence: Sequence, window_size: int) -> int:
    return int((len(sequence) // window_size) * window_size)


class VRHandMYODataset(Dataset):
    def __init__(
        self,
        exps_data: list[dict[str, np.ndarray]],
        window_size: int,
        random_sampling: bool = False,
        samples_per_epoch: Optional = None,
        return_support_info: bool = False,
        transform=None,
        down_sample_target=None,
        use_angles=False,
    ) -> NoReturn:

        self.exps_data = exps_data
        self.window_size = window_size
        self.random_sampling = random_sampling
        self.samples_per_epoch = samples_per_epoch
        self.return_support_info = return_support_info
        self.transform = transform
        self.down_sample_target = down_sample_target
        self.use_angles = use_angles

        if self.random_sampling:
            assert (
                self.samples_per_epoch is not None
            ), "if random_sampling is True samples_per_epoch must be specified"

        # List of ints with strippred lens that shows which frames may and not get into any item
        self._stripped_lens = [
            calc_stripped_len(data["data_vr"], self.window_size) for data in exps_data
        ]
        # Max numbers of different windows without intersections over all data
        self._items_per_stripped_exp = [
            _stripped_len // self.window_size for _stripped_len in self._stripped_lens
        ]

        # Max left idx of window for each exp_data in self.exps_data
        self._max_left_idxs = [
            stripped_len - self.window_size for stripped_len in self._stripped_lens
        ]
        # Probability to choose correspodint exp_data if random_sampling is passed
        self._exp_choose_probas = calc_probas(
            map(lambda x: x["data_vr"], self.exps_data)
        )
        # print('Prob of different moves: ', self._exp_choose_probas)

    def __len__(self) -> int:
        if self.random_sampling:
            return self.samples_per_epoch

        assert sum(self._stripped_lens) % self.window_size == 0
        # Max numbers of different windows without intersections over all data
        max_items = sum(self._items_per_stripped_exp)
        return max_items

    def _window_left_idx_to_data_slice(
        self, exp_data: dict[str, np.ndarray], idx: int
    ) -> dict[str, np.ndarray]:

        return {k: v[idx : idx + self.window_size] for k, v in exp_data.items()}

    def __getitem__(self, idx: int) -> tuple[dict[str, np.ndarray], dict]:

        # Sample random window from random move.
        if idx >= len(self):
            raise IndexError

        if not self.random_sampling:
            running_lens_sum = 0
            for idx_of_exp, max_items in enumerate(self._items_per_stripped_exp):
                running_lens_sum += max_items
                if idx < running_lens_sum:
                    break

            window_idx = idx - (running_lens_sum - max_items)
            window_left_idx = window_idx * self.window_size
            exp_data = self._window_left_idx_to_data_slice(
                self.exps_data[idx_of_exp], window_left_idx
            )

        else:
            idx_of_exp = random.choices(
                range(len(self._exp_choose_probas)),
                self._exp_choose_probas.tolist(),
                k=1,
            )[0]
            window_left_idx = int(random.uniform(0, self._max_left_idxs[idx_of_exp]))
            exp_data = self._window_left_idx_to_data_slice(
                self.exps_data[idx_of_exp], window_left_idx
            )

        # INPUT data
        myo = exp_data["data_myo"].astype("float32")
        myo = rearrange(myo, "t c -> c t")

        if self.transform is not None:
            myo = self.transform(samples=myo, sample_rate=200)

        # TARGET data
        if self.use_angles:
            target = exp_data["data_angles"].astype("float32")
            target = rearrange(target, "t a -> a t")
        else:
            target = exp_data["data_vr"].astype("float32")
            target = rearrange(target, "t b q -> b q t")

        # downsampple target
        if self.down_sample_target is not None:
            target = target[..., :: self.down_sample_target]

        support_info = {
            "idx": idx,
            "idx_of_exp": idx_of_exp,
            "window_left_idx": window_left_idx,
            "len": len(exp_data),
        }

        # assert myo.shape[-1] == vr.shape[-1] == self.window_size, f'{myo.shape}, {vr.shape}, {support_info}'

        if self.return_support_info:
            return myo, target, support_info
        else:
            return myo, target


def fix_sign_quats(data):
    """
    [Times, 16, 4]
    """
    n_times, n_bones, _ = data.shape

    data_tmp = data[:, :, -1].reshape(-1)
    data_sign = np.where(data_tmp < 0, 1.0, -1.0)
    data_sign = data_sign[..., None]

    data_new = data.reshape(-1, 4)
    data_new = data_new * data_sign
    data_new = data_new.reshape(n_times, n_bones, 4)
    return data_new


def create_dataset(
    data_folder: Path,
    original_fps: int,
    delay_ms: int,
    start_crop_ms: int,
    window_size: int,
    random_sampling: bool,
    use_angles=False,
    use_preproc_data=False,
    smooth_h_freq: int = None,
    samples_per_epoch: Optional = None,
    return_support_info: bool = False,
    transform=None,
    down_sample_target=None,
):
    """
    delay_ms - -40 it means emg[40:] and vr[:-40]
    dealy of emg compare with vr. vr changes and we'll see change in emg after 40 ms.
    """
    # Loop over files in data dir
    all_paths = sorted(data_folder.glob("*.npz"))
    all_paths = natsorted(all_paths)

    print(f"Number of moves: {len(all_paths)} | Dataset: {data_folder.parents[1].name}")

    if use_preproc_data:
        exps_data = [dict(np.load(d)) for d in all_paths]

        # temporal alighnment
        # if n_crop_idxs is 0 do nothing
        n_crop_idxs = int(delay_ms / 1000 * original_fps)
        input_keys = ["myo_ts", "data_myo"]

        for i, data in enumerate(exps_data):
            if n_crop_idxs > 0:
                for key, value in data.items():
                    data[key] = (
                        value[:-n_crop_idxs]
                        if key in input_keys
                        else value[n_crop_idxs:]
                    )

            elif n_crop_idxs < 0:
                for key, value in data.items():
                    data[key] = (
                        value[-n_crop_idxs:]
                        if key in input_keys
                        else value[:n_crop_idxs]
                    )

            exps_data[i] = data

        # add reordering of the left type hand
        is_left_hand = check_conditions(["left"], data_folder)
        new_order = [6, 5, 4, 3, 2, 1, 0, 7]
        if is_left_hand:
            for i, data in enumerate(exps_data):
                exps_data[i]["data_myo"] = exps_data[i]["data_myo"][:, new_order]
            print("Reorder this dataset", data_folder, is_left_hand)

        dataset = VRHandMYODataset(
            exps_data,
            window_size=window_size,
            random_sampling=random_sampling,
            samples_per_epoch=samples_per_epoch,
            return_support_info=return_support_info,
            transform=transform,
            down_sample_target=down_sample_target,
            use_angles=use_angles,
        )
        return dataset

    exps_data = []
    for one_exp_data_path in tqdm(all_paths):

        data = load_data_from_one_exp(one_exp_data_path)
        data, left_strip_idx, right_strip_idx = strip_nans_for_one_exp(data)

        if data is None:
            print("No VR for this file:", one_exp_data_path)
            continue

        data = interpolate_quats_for_one_exp(data, quat_interpolate_method="slerp")

        # VLAD: for recordings, that were done with 500 Hz, we use downsampling
        if data["data_myo"].shape[0] > 40_000:
            print(f"Fs before downsampling: {1 / np.mean(np.diff(data['myo_ts']))}")
            data["data_myo"] = data["data_myo"][::2]
            data["data_vr"] = data["data_vr"][::2]
            data["myo_ts"] = data["myo_ts"][::2]
            print(f"Fs after downsampling: {1 / np.mean(np.diff(data['myo_ts']))}")

        # VLAD: filtering of raw data
        empty = np.empty(data["data_myo"].shape)

        std_range = 6

        for electrode in range(data["data_myo"].shape[1]):
            da = data["data_myo"][:, electrode]
            dat = da[~np.isnan(da)]

            samp_freq = 250  # Sample frequency (Hz)
            notch_freq = 50.0  # Frequency to be removed from signal (Hz)
            quality_factor = 30.0  # Quality factor

            b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
            freq, h = signal.freqz(b_notch, a_notch, fs=samp_freq)

            emg_filt = signal.filtfilt(b_notch, a_notch, dat)

            emg_filt_norm = butter_highpass_filter(emg_filt, 10, 125, order=5)

            diff_in_length = len(da) - len(emg_filt_norm)

            emg_filt_norm = np.append(emg_filt_norm, np.zeros(diff_in_length) + np.nan)

            empty[:, electrode] = emg_filt_norm

        # cutting down outliers
        cleanest = np.empty(data["data_myo"].shape)
        std_threshold = std_range * np.max(np.nanstd(empty, axis=0))

        for electrode in range(data["data_myo"].shape[1]):
            one_signal = empty[:, electrode]
            one_signal_clean = one_signal[~np.isnan(one_signal)]

            clear_up = np.where(
                one_signal_clean < std_threshold, one_signal_clean, std_threshold
            )
            clear_down = np.where(
                clear_up > -1 * std_threshold, clear_up, -1 * std_threshold
            )

            diff_in_length = len(one_signal) - len(clear_down)
            final_signal = np.append(clear_down, np.zeros(diff_in_length) + np.nan)
            cleanest[:, electrode] = final_signal

        maxx = np.nanmax(cleanest)
        minn = np.nanmin(cleanest)
        # here I (Vlad) normalize data based on the whole set of signals.
        # EMG preproc: normalize -> (-1, 1) range as audio.

        emg_min_max = (cleanest - minn) / (maxx - minn)  # (0, 1)
        emg_min_max = 2 * emg_min_max - 1

        # I do scaling on the whole dataset, but that creates vertical shifts (+- 0.2) in data, so i substract this shift
        data_myo = emg_min_max - np.nanmean(emg_min_max, axis=0)

        data["data_myo"] = data_myo

        # VR quats preproc:
        data["data_vr"] = data["data_vr"][:, :, 4:8]
        data["data_vr"] = np.stack([inverse_rotations(r) for r in data["data_vr"]])
        data["data_vr"] = fix_sign_quats(data["data_vr"])

        # Crop starting bad points
        start_crop_idxs = int(start_crop_ms / 1000 * original_fps)
        n_crop_idxs = int(delay_ms / 1000 * original_fps)

        data["data_vr"] = data["data_vr"][start_crop_idxs:]
        data["data_myo"] = data["data_myo"][start_crop_idxs:]
        data["myo_ts"] = data["myo_ts"][start_crop_idxs:]

        # temporal alighnment
        # if n_crop_idxs is 0 do nothing
        if n_crop_idxs > 0:
            data["data_vr"] = data["data_vr"][n_crop_idxs:]
            data["data_myo"] = data["data_myo"][:-n_crop_idxs]
        elif n_crop_idxs < 0:
            data["data_vr"] = data["data_vr"][:n_crop_idxs]
            data["data_myo"] = data["data_myo"][-n_crop_idxs:]

        assert (
            data["data_vr"].shape[0] == data["data_myo"].shape[0]
        ), f'lens of data_vr and data_myo are different {data["data_vr"].shape} !=  {data["data_myo"].shape}'

        exps_data.append(data)

    dataset = VRHandMYODataset(
        exps_data,
        window_size=window_size,
        random_sampling=random_sampling,
        samples_per_epoch=samples_per_epoch,
        return_support_info=return_support_info,
        transform=transform,
        down_sample_target=down_sample_target,
    )

    print(f"Total len: {len(dataset)}")  # max numbers of different windows over

    return dataset


def prepare_data(
    data_folder: Path,
    original_fps: int,
    delay_ms: int,
    start_crop_ms: int,
    window_size: int,
    random_sampling: bool,
    samples_per_epoch: Optional = None,
    return_support_info: bool = False,
    transform=None,
    down_sample_target=None,
):
    """
    delay_ms - -40 it means emg[40:] and vr[:-40]
    dealy of emg compare with vr. vr changes and we'll see change in emg after 40 ms.
    """
    # Loop over files in data dir
    all_paths = sorted(data_folder.glob("*.npz"))
    all_paths = natsorted(all_paths)

    print(f"Number of moves: {len(all_paths)} | Dataset: {data_folder.parents[1].name}")

    exps_data = []
    for one_exp_data_path in tqdm(all_paths):

        data = load_data_from_one_exp(one_exp_data_path)
        data, _, _ = strip_nans_for_one_exp(data)

        if data is None:
            print("No VR for this file:", one_exp_data_path)
            continue

        data = interpolate_quats_for_one_exp(data, quat_interpolate_method="slerp")

        # VLAD: for recordings, that were done with 500 Hz, we use downsampling
        if data["data_myo"].shape[0] > 40_000:
            print(f"Fs before downsampling: {1 / np.mean(np.diff(data['myo_ts']))}")
            data["data_myo"] = data["data_myo"][::2]
            data["data_vr"] = data["data_vr"][::2]
            data["myo_ts"] = data["myo_ts"][::2]
            print(f"Fs after downsampling: {1 / np.mean(np.diff(data['myo_ts']))}")

        # VLAD: filtering of raw data
        empty = np.empty(data["data_myo"].shape)

        std_range = 6

        for electrode in range(data["data_myo"].shape[1]):
            da = data["data_myo"][:, electrode]
            dat = da[~np.isnan(da)]

            samp_freq = 250  # Sample frequency (Hz)
            notch_freq = 50.0  # Frequency to be removed from signal (Hz)
            quality_factor = 30.0  # Quality factor

            b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
            freq, h = signal.freqz(b_notch, a_notch, fs=samp_freq)

            emg_filt = signal.filtfilt(b_notch, a_notch, dat)

            emg_filt_norm = butter_highpass_filter(emg_filt, 10, 125, order=5)

            diff_in_length = len(da) - len(emg_filt_norm)

            emg_filt_norm = np.append(emg_filt_norm, np.zeros(diff_in_length) + np.nan)

            empty[:, electrode] = emg_filt_norm

        # cutting down outliers
        cleanest = np.empty(data["data_myo"].shape)
        std_threshold = std_range * np.max(np.nanstd(empty, axis=0))

        for electrode in range(data["data_myo"].shape[1]):
            one_signal = empty[:, electrode]
            one_signal_clean = one_signal[~np.isnan(one_signal)]

            clear_up = np.where(
                one_signal_clean < std_threshold, one_signal_clean, std_threshold
            )
            clear_down = np.where(
                clear_up > -1 * std_threshold, clear_up, -1 * std_threshold
            )

            diff_in_length = len(one_signal) - len(clear_down)
            final_signal = np.append(clear_down, np.zeros(diff_in_length) + np.nan)
            cleanest[:, electrode] = final_signal

        maxx = np.nanmax(cleanest)
        minn = np.nanmin(cleanest)
        # here I (Vlad) normalize data based on the whole set of signals.
        # EMG preproc: normalize -> (-1, 1) range as audio.

        emg_min_max = (cleanest - minn) / (maxx - minn)  # (0, 1)
        emg_min_max = 2 * emg_min_max - 1

        # I do scaling on the whole dataset, but that creates vertical shifts (+- 0.2) in data, so i substract this shift
        data_myo = emg_min_max - np.nanmean(emg_min_max, axis=0)

        data["data_myo"] = data_myo

        # VR quats preproc:
        data["data_vr"] = data["data_vr"][:, :, 4:8]
        data["data_vr"] = np.stack([inverse_rotations(r) for r in data["data_vr"]])
        data["data_vr"] = fix_sign_quats(data["data_vr"])

        # Crop starting bad points
        start_crop_idxs = int(start_crop_ms / 1000 * original_fps)
        n_crop_idxs = int(delay_ms / 1000 * original_fps)

        data["data_vr"] = data["data_vr"][start_crop_idxs:]
        data["data_myo"] = data["data_myo"][start_crop_idxs:]
        data["myo_ts"] = data["myo_ts"][start_crop_idxs:]

        # temporal alighnment
        # if n_crop_idxs is 0 do nothing
        if n_crop_idxs > 0:
            data["data_vr"] = data["data_vr"][n_crop_idxs:]
            data["data_myo"] = data["data_myo"][:-n_crop_idxs]
        elif n_crop_idxs < 0:
            data["data_vr"] = data["data_vr"][:n_crop_idxs]
            data["data_myo"] = data["data_myo"][-n_crop_idxs:]

        assert (
            data["data_vr"].shape[0] == data["data_myo"].shape[0]
        ), f'lens of data_vr and data_myo are different {data["data_vr"].shape} !=  {data["data_myo"].shape}'

        exps_data.append(data)

    dataset = VRHandMYODataset(
        exps_data,
        window_size=window_size,
        random_sampling=random_sampling,
        samples_per_epoch=samples_per_epoch,
        return_support_info=return_support_info,
        transform=transform,
        down_sample_target=down_sample_target,
    )

    print(f"Total len: {len(dataset)}")  # max numbers of different windows over

    return dataset        

# augmentations
def make_electrode_shifting(data, min_angle, max_angle, p=0.5):
    """
    Apply rotation of the sensors by one angle.

    Also we can aplly some small pertubration for each sensors.
    Also it migh be interestin to use temporal inforamtion and apply
    distraction in spatial and temporal dimension.

    Rotate 8 electrodes by circle with rollover.
    min_angle - in degree
    max_angle - maximum
    prob

    data [8, N]

    """

    # print('WWW', data.shape)
    n_sensors = data.shape[0]
    angles = np.linspace(0, 360, n_sensors + 1)[:-1]

    # calculate random addition angle.
    # angle_diffs = np.random.choice([-1,1], size=n_sensors) * np.random.uniform(low=min_angle, high=max_angle, size = n_sensors)
    # new_angles = np.array([phi +delta  for phi, delta in zip(angles, angle_diffs)])

    # just only one angle for rotation.
    delta = np.random.choice([-1, 1]) * np.random.uniform(low=min_angle, high=max_angle)
    new_angles = np.array([phi + delta for phi in angles])

    new_angles = np.where(new_angles < 0, new_angles + 360, new_angles)
    new_angles = np.where(new_angles > 360, new_angles - 360, new_angles)

    # interpoaltion
    # add 360 degree for interpoaltion.
    x_ = np.append(angles, 360.0)
    data_expand = np.concatenate([data, data[:1]])

    f1 = interp1d(x_, data_expand, kind="cubic", axis=0)
    res = f1(new_angles)

    return res

    # def apply_transform():


class SpatialRotation(BaseWaveformTransform):
    """
    Apply a constant amount of gain, so that highest signal level present in the sound becomes
    0 dBFS, i.e. the loudest level allowed if all samples must be between -1 and 1. Also known
    as peak normalization.
    """

    supports_multichannel = True

    def __init__(self, min_angle, max_angle, p=0.5):

        super().__init__(p)
        self.min_angle = min_angle
        self.max_angle = max_angle

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)

    def apply(self, samples, sample_rate):
        result = make_electrode_shifting(samples, self.min_angle, self.max_angle)

        return result
