import torch
import torch.utils.data as data
import numpy as np
import os
from typing import *
import random
import scipy.stats as stats
from rich.progress import track
from rich.status import Status

class XianTrajectoryDataset(data.Dataset):
    Unix_time_20161001 = 1475276400
    seconds_per_day = 86400
    minutes_per_day = 1440
    Unix_time_20161101 = 1477958400

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __init__(self, dataset_root: str, traj_length: int, feature_mean: List[float], feature_std: List[float],
                 city_cells: int = 256):
        """
        Didi Trajectory Dataset Xi'an datasets

        The processed dataset folder contains many gps_YYYYMMDD.pt files
        They record trajectories of all orders in that day, formatted as:

        [
            (driver_id, order_id, lon_lat_tensor, time_tensor),
            (driver_id, order_id, lon_lat_tensor, time_tensor),
            ...
        ]

        lon_lat_tensor: (N, 2) torch.float64
        time_tensor: (N,) torch.long

        Otherwise, if only cache is available
        Set dataset_root to "nov" so that the class correctly recognizes the time
        Then use loadCache() to load the cache

        :param dataset_root: The path to the folder containing gps_YYYYMMDD.pt files
        :param traj_length: shorter: not included, longer: included but cropped
        :param feature_mean: The mean of time&lon&lat
        :param feature_std: The std of time&lon&lat
        """

        if "oct" in dataset_root:
            self.time_shift = self.Unix_time_20161001
        elif "nov" in dataset_root:
            self.time_shift = self.Unix_time_20161101
        else:
            raise ValueError(f"dataset_root should contain 'oct' or 'nov', but got {dataset_root}")

        self.traj_mean = torch.tensor(feature_mean, dtype=torch.float32).view(1, 3)
        self.traj_std = torch.tensor(feature_std, dtype=torch.float32).view(1, 3)

        self.dataset_root = dataset_root
        self.file_paths = [os.path.join(dataset_root, file) for file in os.listdir(dataset_root) if
                           file.endswith('.pt')]
        self.part_idx = 0

        self.traj_length = traj_length
        self.sample_length = traj_length
        self.erase_rate = 0.0
        self.shift_rate = 0.0

        self.dataset_part = []

        self.city_cell_boundaries = self.__getGaussianSegments(city_cells)


    def resetSampleLength(self, length: int) -> None:
        """
        Reset the sample length of trajectory
        :param length: the new sample length
        :return: None
        """
        if length > self.traj_length:
            length = self.traj_length
        elif length < 1:
            length = 1
        self.sample_length = length


    def resetEraseRate(self, elim_rate: float) -> None:
        """
        Reset the erase rate of trajectory
        :param elim_rate: The percentage of points to be erased
        :return: None
        """
        self.erase_rate = min(1.0, max(0.0, elim_rate))


    def resetShiftRate(self, shift_rate: float) -> None:
        """
        Never used, set percentage of points to be randomly shifted
        """
        self.shift_rate = min(1.0, max(0.0, shift_rate))


    def loadNextFiles(self, load_n: int) -> bool:
        """
        Load next n files into memory
        Not necessary once the cache is saved
        :return: True if there are still files to load, False if all files are loaded
        """
        # First clear the previous dataset_part
        self.dataset_part = []
        for i in range(load_n):
            if self.part_idx >= len(self.file_paths):
                self.part_idx = 0
                break
            file_path = self.file_paths[self.part_idx]
            with Status(f'Loading {file_path} from disk...'):
                raw_dataset_part = torch.load(file_path)

            for (_, _, lon_lat_tensor, time_tensor) in track(raw_dataset_part, description=f'Parsing {file_path}'):
                if lon_lat_tensor.shape[0] >= self.traj_length:
                    traj = lon_lat_tensor[:self.traj_length].to(torch.float32)
                    traj = ((traj - self.traj_mean[:, 1:]) / self.traj_std[:, 1:]).transpose(0, 1).contiguous()
                    # traj: (2, N) torch.float32

                    # minutes since month start
                    times = ((time_tensor[:self.traj_length] - self.time_shift) / 60).to(torch.float32)
                    # minutes of the day of trajectory beginning, since month start
                    begin_of_the_day = torch.floor(times[0] / self.minutes_per_day) * self.minutes_per_day
                    # For a trajectory point, its time has range 0 - 1, corresponds to the period of a day
                    # However, in case if a trajectory across multiple days, the time will be like this:
                    # {..., 0.98, 0.99, 0.00, 0.01, 0.02, ...}, notice 0.99 to 0.00 just cross midnight 00:00
                    # This causes error in linear interpolation, and also makes the trajectory time unsorted
                    # So, we instead make times {..., 0.98, 0.99, 1.00, 1.01, 1.02}
                    # Only the first time is guaranteed in range 0 to 1
                    traj_daytime = (times - begin_of_the_day) / self.minutes_per_day

                    lon_lat_t = torch.cat([traj, traj_daytime.view(1, -1)], dim=0)  # (3, N)

                    equalized_traj = self.sampleRateEqualize(lon_lat_t)

                    self.dataset_part.append((lon_lat_t, equalized_traj))

            self.part_idx += 1

        return len(self.dataset_part) > 0


    def loadFile(self, file_path: str):
        """
        Load a specific file into memory
        """
        # First clear the previous dataset_part
        self.dataset_part = []
        print(f'Loading {file_path}')
        raw_dataset_part = torch.load(file_path)

        for (_, _, lon_lat_tensor, time_tensor) in raw_dataset_part:
            if lon_lat_tensor.shape[0] >= self.traj_length:
                traj = lon_lat_tensor[:self.traj_length].to(torch.float32)
                traj = ((traj - self.traj_mean[:, 1:]) / self.traj_std[:, 1:]).transpose(0, 1).contiguous()
                self.dataset_part.append((
                    traj,
                    ((time_tensor[:self.traj_length] - self.time_shift) / 60).to(torch.float32),
                ))


    def __getGaussianSegments(self, n_segments: int) -> torch.Tensor:
        """
        It is used to divide city grids, but in a gaussian way
        Closer to the center, the grid is finer, further from the center, the grid is coarser
        :param n_segments: The number of segments
        :return: The boundaries of the segments
        """
        mean = 0
        std_dev = 1
        gaussian_dist = stats.norm(mean, std_dev)

        # Divide the range of the Gaussian distribution into N segments
        segment_area = 1 / n_segments
        segment_boundaries = [0]
        segment_areas = np.zeros(n_segments)

        for i in range(n_segments):
            # Find the x value that corresponds to the segment area
            x = gaussian_dist.ppf(segment_area * (i + 1))
            segment_boundaries.append(x)

            # Find the area under the curve of the segment
            segment_areas[i] = gaussian_dist.cdf(x) - gaussian_dist.cdf(segment_boundaries[i])

        segment_boundaries[0] = -np.inf
        segment_boundaries[-1] = np.inf

        return torch.Tensor(segment_boundaries).to(self.device)


    def shuffleAllFiles(self):
        """
        Never used, shuffle all files
        :return:
        """
        if self.part_idx != 0:
            raise RuntimeError('You should call loadNextFiles() to load all files before shuffling')
        random.shuffle(self.file_paths)


    def __len__(self) -> int:
        return len(self.dataset_part)


    def getCellIndex(self, point: torch.Tensor) -> torch.Tensor:
        """ get city cell index of a point

        :param point: (2,) tensor, (lon, lat)
        :return: (2,) tensor, (lon_idx, lat_idx)
        """
        if point.ndim == 1:
            lon_idx = torch.searchsorted(self.city_cell_boundaries, point[0]) - 1
            lat_idx = torch.searchsorted(self.city_cell_boundaries, point[1]) - 1
            return torch.tensor([lon_idx, lat_idx], dtype=torch.long, device=self.device)
        elif point.ndim == 2:
            # point: (N, 2)
            lon_idx = torch.searchsorted(self.city_cell_boundaries, point[:, 0]) - 1  # (N,)
            lat_idx = torch.searchsorted(self.city_cell_boundaries, point[:, 1]) - 1  # (N,)
            result = torch.stack([lon_idx, lat_idx], dim=1)  # (N, 2)
            # eliminate consecutive duplicates
            result[1:] = result[1:] * (result[1:] != result[:-1]).all(dim=1, keepdim=True)
            return result


    def sampleRateEqualize(self, original_traj: torch.Tensor) -> torch.Tensor:
        """
        To make points have equal distance, using linear interpolation.
        First up sample the trajectory 10x denser with linear interpolation,
        then use this overly dense trajectory to sample the equalized trajectory
        :param original_traj: the trajectory to be equalized, (3, L)
        :return: the equalized trajectory, (3, L)
        """
        original_traj = original_traj.to("cuda")
        L = original_traj.shape[1]

        # --- STEP 1. compute cumulative distance of the trajectory ---
        seg_lengths = torch.sqrt(torch.sum((original_traj[:2, 1:] - original_traj[:2, :-1]) ** 2, dim=0))  # (L-1,)
        traj_dist = float(torch.sum(seg_lengths))
        cum_seg_lengths = torch.cat([torch.zeros(1, device="cuda"), torch.cumsum(seg_lengths, dim=0)], dim=0)  # (L,)

        # --- STEP 2. over sample the cumulative distance, generate sample pivots ---
        # over_sample_lens: (10*L,)
        over_sample_lens = torch.nn.functional.interpolate(
            cum_seg_lengths.view(1, 1, -1), scale_factor=10, mode='linear', align_corners=True).view(-1)
        # sample_pivots: (L,), it is the place where we should sample a point
        sample_pivots = torch.linspace(0, traj_dist, L, device="cuda")

        # --- STEP 3. assign each sample pivot to a point in over_sample_traj ---
        # compute dist between every point in over_sample_traj and sample_pivots
        # distances: (10*L, L) = (10*L, 1) - (1, L)
        distances = torch.abs(over_sample_lens.unsqueeze(1) - sample_pivots.unsqueeze(0))
        assignments = torch.argmin(distances, dim=0)    # (L,)
        over_sample_traj = torch.nn.functional.interpolate(
            original_traj.unsqueeze(0), scale_factor=10, mode='linear', align_corners=True).squeeze(0)

        result = torch.zeros_like(original_traj)
        result[:, 0] = original_traj[:, 0]
        result[:, -1] = original_traj[:, -1]
        result[:, 1:-1] = over_sample_traj[:, assignments[1:-1]]

        return result.to("cpu")



    def __getitem__(self, index: int) -> Any:
        """
        :param index: The index of the trajectory in dataset_part
        :return:
            traj: the original trajectory, (3, L)
            eq_traj: the equalized trajectory, (3, L)
            insertion_mask: insertion_mask[i] = 3 means 3 new points should be generated and inserted to the
                traj !!After Erase!! between point i and point i+1, shape: (L,)
            numeric_attr: the numeric attributes, (2,)
            categorical_attr: the categorical attributes, (4,)
            binary_mask: binary_mask[i] = 1 if point i in traj should be erased, 0 otherwise, (L,)
            n_erased: the number of erased points, (1,)
        """
        # lon_lat: (N, 3), times: (N,)
        traj, eq_traj = self.dataset_part[index]
        sample_start = random.randint(0, self.traj_length - self.sample_length)
        traj = traj[:, sample_start:sample_start + self.sample_length].to(self.device)
        eq_traj = eq_traj[:, sample_start:sample_start + self.sample_length].to(self.device)


        # Get the length of the trajectory
        traj_length = torch.sqrt(torch.sum((traj[:2, 1:] - traj[:2, :-1]) ** 2, dim=1)).sum()  # the length of the trajectory

        # Get the average move distance
        avg_move_distance = traj_length / (traj.shape[1] - 1)

        # Get the cell index of the start and end points
        # The length of cell_indices is variable, so cannot be packed into a tensor
        # cell_indices = self.getCellIndex(lon_lat)  # (2, M)
        numeric_attr = torch.tensor([traj_length, avg_move_distance]).to(torch.float32).to(self.device)  # (3,)

        n_remain = self.sample_length - int(self.sample_length * self.erase_rate)
        n_erased = self.sample_length - n_remain
        # select n_remain indices to remain, sorted, and exclude the first and last point
        remain_indices = torch.randperm(self.sample_length - 2)[:n_remain - 2].to(self.device) + 1
        remain_indices = torch.sort(remain_indices)[0]
        # add firsst and the last point
        remain_indices = torch.cat([torch.tensor([0], device=self.device), remain_indices,
                                    torch.tensor([self.sample_length - 1], device=self.device)])
        # insert_mask is 0 if the point is kept, 1 if the point is erased
        # in other words, insert_mask is 1 if this position requires insertion
        insertion_mask = remain_indices[1:] - remain_indices[:-1] - 1
        binary_mask = torch.ones(self.sample_length, dtype=torch.float32, device=self.device)
        binary_mask[remain_indices] = 0

        src_cell_idx = self.getCellIndex(traj[:2, 0])  # (2,)
        dst_cell_idx = self.getCellIndex(traj[:2, -1])  # (2,)
        categorical_attr = torch.cat([src_cell_idx, dst_cell_idx], dim=0).to(torch.long)  # (5,)

        # traj: original trajetcory
        # eq_traj: traj with unit sample intervals
        # insertion_mask: [i] = N represents N points should be inserted between i and i+1
        # numeric_attr: [traj_length, avg_move_distance]
        # binary_mask: 1 if the place is erased, 0 if it is remained
        return traj, eq_traj, insertion_mask, numeric_attr, categorical_attr, binary_mask, n_erased


    @property
    def n_files(self) -> int:
        return len(self.file_paths)


    def saveCache(self, path: str):
        torch.save(self.dataset_part, path)


    def loadCache(self, path: str):
        with Status(f'Loading {path} from disk...'):
            self.dataset_part = torch.load(path)


def collectFunc(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    """
    :param batch: A list of trajectories and attributes
    :return: lon_lat_t: (B, 3, N), cell_indices:
    """
    (traj_0_list, eq_traj_list, insertion_mask_list, num_attr_list, cat_attr_list, bin_mask_list, n_erased_list) = zip(*batch)
    traj_0 = torch.stack(traj_0_list, dim=0)
    traj_eq = torch.stack(eq_traj_list, dim=0)
    B, _, L = traj_0.shape
    insertion_masks = torch.stack(insertion_mask_list, dim=0)
    num_attrs = torch.stack(num_attr_list, dim=0)
    cat_attrs = torch.stack(cat_attr_list, dim=0)
    bin_masks = torch.stack(bin_mask_list, dim=0)
    n_erased = torch.tensor(n_erased_list, device="cuda", dtype=torch.long)

    return traj_0, traj_eq, n_erased, bin_masks.unsqueeze(1), num_attrs, cat_attrs, insertion_masks


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    xian_nov_dataset_args = {
        "dataset_root": "E:/Data/Didi/xian/nov",
        "traj_length": 256,
        "feature_mean": [21599.4980, 108.950773428688, 34.24354179925547],    # time lon lat
        "feature_std": [12470.9102, 0.02129110045580343, 0.019358855648211895],
        "city_cells": 64,
    }
    dataset = DidiTrajectoryDataset(**xian_nov_dataset_args)
    # dataset.loadNextFiles(10)
    # dataset.saveCache("Dataset/10_days_cache.pth")
    dataset.loadCache("Dataset/2_days_cache.pth")

    loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0, collate_fn=collectFunc)

    loader.dataset.resetEraseRate(0.4)
    loader.dataset.resetShiftRate(0.02)

    traj_erase, traj_broken, attrs, traj_0, traj_eq, insertion_masks, traj_0_len, bin_mask, insert_traj = next(iter(loader))

    print(f"traj_broken.shape: {traj_broken.shape}")
    print(f"numeric_attr.shape: {attrs.shape}")
    print(f"traj_0.shape: {traj_0.shape}")
    print(f"target_n_points: {traj_0_len}")

    for i in track(range(5), description="Plotting"):
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.title(f"Broken Trajectory |T|={traj_broken.shape[2]}")
        plt.plot(traj_broken[i, 0].cpu().detach(), traj_broken[i, 1].cpu().detach(), 'b-', linewidth=0.5)
        plt.scatter(traj_broken[i, 0].cpu().detach(), traj_broken[i, 1].cpu().detach(), s=2)

        plt.subplot(2, 2, 2)
        plt.title(f"Target Trajectory |T|={traj_0.shape[2]}")
        plt.plot(traj_0[i, 0].cpu().detach(), traj_0[i, 1].cpu().detach(), 'b-', linewidth=0.5)
        plt.scatter(traj_0[i, 0].cpu().detach(), traj_0[i, 1].cpu().detach(), s=2)

        # do z-score normalization
        traj_mean = torch.mean(traj_0[i, :, :], dim=1, keepdim=True)
        traj_std = torch.std(traj_0[i, :, :], dim=1, keepdim=True)
        norm_traj = (traj_0[i, :, :] - traj_mean) / traj_std

        segment_lengths = torch.sqrt(torch.sum((norm_traj[:2, 1:] - norm_traj[:2, :-1]) ** 2, dim=0))
        length_mean = torch.mean(segment_lengths)
        length_std = torch.std(segment_lengths)
        length_range = torch.max(segment_lengths) - torch.min(segment_lengths)
        plt.subplot(2, 2, 3)
        plt.title(f"Segment Lengths")
        plt.bar(range(len(segment_lengths)), segment_lengths.cpu().detach().numpy())
        text = f"mean: {length_mean:.5f}\nstd: {length_std:.5f}\nrange: {length_range:.5f}"
        # put text on the top of the bar
        plt.text(0.5, 0.95, text, horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)


        plt.subplot(2, 2, 4)
        plt.title(f"Equalized Trajectory |T|={traj_eq.shape[2]}")
        plt.plot(traj_eq[i, 0].cpu().detach(), traj_eq[i, 1].cpu().detach(), 'b-', linewidth=0.5)
        plt.scatter(traj_eq[i, 0].cpu().detach(), traj_eq[i, 1].cpu().detach(), s=2)

        plt.tight_layout()

        plt.show()

        # plt.savefig(f"Dataset/TrajVisualize/fig_{i}.png", dpi=300)
        #
        # plt.close()
