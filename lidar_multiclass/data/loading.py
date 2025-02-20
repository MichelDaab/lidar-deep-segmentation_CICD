"""How to represent points clouds, specifying specific data signatures (e.g. choice of features).

Country-specific definitions inherit from abstract class LidarDataLogic, which implements general logics
to split each point cloud into subtiles, format these subtiles, and save a learning-ready dataset splitted into
train, val. A test set is also created, which is simply a copy of the selected test point clouds. This is so
test phase conditions are similar to "in-the-wild" prediction conditions. 

In particular, subclasses implement a "load_las" method that is used by the datamodule at test and inference time.

"""

from abc import ABC, abstractmethod
import argparse
import os, glob
import os.path as osp
from shutil import copyfile
from tqdm import tqdm
import laspy
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


class LidarDataLogic(ABC):
    """Abstract class to load, chunk, and save a point cloud dataset according to a train/val/test split.
    load_las and its needed parameters ares specified in child classes.

    """

    split = ["val", "train", "test"]
    input_tile_width_meters = 1000
    subtile_width_meters = 50
    return_num_normalization_max_value = 7

    def __init__(self, **kwargs):
        self.input_data_dir = kwargs.get("input_data_dir")
        self.prepared_data_dir = kwargs.get("prepared_data_dir")
        self.split_csv = kwargs.get("split_csv")
        self.range_by_axis = np.arange(
            self.input_tile_width_meters // self.subtile_width_meters + 1
        )

    @abstractmethod
    def load_las(self, las_filepath: str) -> Data:
        """Load a point cloud in LAS format to memory and turn it into torch-geometric Data object.

        Args:
            las_filepath (str): path to the LAS file.

        Returns:
            Data: The point cloud formatted for later deep learning training.

        """
        raise NotImplementedError

    def prepare(self):
        """Prepare a dataset for model training and model evaluation.

        Iterates through LAS files listed in a csv metadata file. A `split` column
        specifies the train/val/test split of the dataset to be created.
        Depending on the set, this method will:

        train/val:
            Load LAS into memory as a Data object with selected features,
            then iteratively extract 50m*50m subtiles by filtering along x
            then y axis. Serialize the resulting Data object using torch.save.

        test:
            Simply copy the LAS to the new test folder.

        """
        split_df = pd.read_csv(self.split_csv)
        for phase in tqdm(self.split, desc="Phases"):
            basenames = split_df[split_df.split == phase].basename.tolist()
            print(f"Subset: {phase}")
            print("  -  ".join(basenames))
            for file_basename in tqdm(basenames, desc="Files"):
                filepath = self._find_file_in_dir(self.input_data_dir, file_basename)
                output_subdir_path = osp.join(self.prepared_data_dir, phase)
                if phase == "test":
                    os.makedirs(output_subdir_path, exist_ok=True)
                    target_file = osp.join(output_subdir_path, file_basename)
                    copyfile(filepath, target_file)
                elif phase in ["train", "val"]:
                    output_subdir_path = osp.join(
                        output_subdir_path, osp.basename(filepath)
                    )
                    os.makedirs(output_subdir_path, exist_ok=True)
                    self.split_and_save(filepath, output_subdir_path)
                else:
                    raise KeyError("Phase should be one of train/val/test.")

    def split_and_save(self, filepath: str, output_subdir_path: str) -> None:
        """Parse a LAS, extract and save each subtile as a Data object.

        Args:
            filepath (str): input LAS file
            output_subdir_path (str): output directory to save splitted `.data` objects.
        """
        data = self.load_las(filepath)
        idx = 0
        for _ in tqdm(self.range_by_axis):
            if len(data.pos) == 0:
                break
            data_x_band = self._extract_by_x(data)
            for _ in self.range_by_axis:
                if len(data_x_band.pos) == 0:
                    break
                subtile_data = self._extract_by_y(data_x_band)
                self._save(subtile_data, output_subdir_path, idx)
                idx += 1

    def _find_file_in_dir(self, input_data_dir: str, basename: str) -> str:
        """Query files with .las extension in subfolder of input_data_dir.

        Args:
            input_data_dir (str): data directory

        Returns:
            [str]: first file path matching the query.

        """
        query = f"{input_data_dir}*{basename}"
        files = glob.glob(query)
        return files[0]

    def _extract_by_axis(self, data: Data, axis=0) -> Data:
        """Filter a data object on a chosen axis, using a relative position .
        Modifies the original data object so that extracted future filters are faster.

        Args:
            data (Data): a pyg Data object with pos, x, and y attributes.
            relative_pos (int): where the data to extract start on chosen axis (typically in range 0-1000)
            axis (int, optional): 0 for x and 1 for y axis. Defaults to 0.

        Returns:
            Data: the data that is at most subtile_width_meters above relative_pos on the chosen axis.
        """
        sub_tile_data = data.clone()
        pos_axis = sub_tile_data.pos[:, axis]
        absolute_low = pos_axis.min(0)
        absolute_high = absolute_low + self.subtile_width_meters
        mask = (absolute_low <= pos_axis) & (pos_axis <= absolute_high)

        # select
        sub_tile_data.pos = sub_tile_data.pos[mask]
        sub_tile_data.x = sub_tile_data.x[mask]
        sub_tile_data.y = sub_tile_data.y[mask]

        data.pos = data.pos[~mask]
        data.x = data.x[~mask]
        data.y = data.y[~mask]
        return sub_tile_data

    def _extract_by_x(self, data: Data) -> Data:
        """extract_by_axis applied on first axis x"""
        return self._extract_by_axis(data, axis=0)

    def _extract_by_y(self, data: Data) -> Data:
        """extract_by_axis applied on second axis y"""
        return self._extract_by_axis(data, axis=1)

    def _save(self, subtile_data: Data, output_subdir_path: str, idx: int) -> None:
        """Save the subtile data object with torch.

        Args:
            subtile_data (Data): the object to save.
            output_subdir_path (str): the subfolder to save it.
            idx (int): an arbitrary but unique subtile identifier.
        """
        subtile_save_path = osp.join(output_subdir_path, f"{str(idx).zfill(4)}.data")
        torch.save(subtile_data, subtile_save_path)


class FrenchLidarDataLogic(LidarDataLogic):

    x_features_names = [
        "intensity",
        "return_num",
        "num_returns",
        "red",
        "green",
        "blue",
        "nir",
        "rgb_avg",
        "ndvi",
    ]
    colors_normalization_max_value = 255 * 256

    @classmethod
    def load_las(self, las_filepath: str):
        f"""Loads a point cloud in LAS format to memory and turns it into torch-geometric Data object.

        Builds a composite (average) color channel on the fly.

        Calculate NDVI on the fly.

        x_features_names are {' - '.join(self.x_features_names)}

        Args:
            las_filepath (str): path to the LAS file.

        Returns:
            Data: the point cloud formatted for later deep learning training.

        """
        las = laspy.read(las_filepath)
        pos = np.asarray([las.x, las.y, las.z], dtype=np.float32).transpose()

        x = np.asarray(
            [
                las[x_name]
                for x_name in [
                    "intensity",
                    "return_num",
                    "num_returns",
                    "red",
                    "green",
                    "blue",
                    "nir",
                ]
            ],
            dtype=np.float32,
        ).transpose()

        return_num_idx = self.x_features_names.index("return_num")
        occluded_points = x[:, return_num_idx] > 1

        x[:, return_num_idx] = (x[:, return_num_idx]) / (
            self.return_num_normalization_max_value
        )
        num_return_idx = self.x_features_names.index("num_returns")
        x[:, num_return_idx] = (x[:, num_return_idx]) / (
            self.return_num_normalization_max_value
        )

        for idx, c in enumerate(self.x_features_names):
            if c in ["red", "green", "blue", "nir"]:
                assert x[:, idx].max() <= self.colors_normalization_max_value
                x[:, idx] = x[:, idx] / self.colors_normalization_max_value
                x[occluded_points, idx] = 0

        red = x[:, self.x_features_names.index("red")]
        green = x[:, self.x_features_names.index("green")]
        blue = x[:, self.x_features_names.index("blue")]

        rgb_avg = np.asarray([red, green, blue], dtype=np.float32).mean(axis=0)

        nir = x[:, self.x_features_names.index("nir")]
        ndvi = (nir - red) / (nir + red + 10**-6)
        x = np.concatenate([x, rgb_avg[:, None], ndvi[:, None]], axis=1)

        try:
            # for LAS format V1.2
            y = las.classification.array.astype(int)
        except:
            # for  LAS format V1.4
            y = las.classification.astype(int)

        return Data(
            pos=pos,
            x=x,
            y=y,
            las_filepath=las_filepath,
            x_features_names=self.x_features_names,
        )


class SwissTopoLidarDataLogic(LidarDataLogic):
    x_features_names = [
        "intensity",
        "return_num",
        "num_returns",
        "red",
        "green",
        "blue",
        "rgb_avg",
    ]
    colors_normalization_max_value = 256

    @classmethod
    def load_las(self, las_filepath: str) -> Data:
        """Loads a point cloud in LAS format to memory and turns it into torch-geometric Data object.

        Builds a composite (average) color channel on the fly.

        Args:
            las_filepath (str): path to the LAS file.

        Returns:
            Data: the point cloud formatted for later deep learning training.

        """
        las = laspy.read(las_filepath)
        pos = np.asarray([las.x, las.y, las.z], dtype=np.float32).transpose()

        x = np.asarray(
            [
                las[x_name]
                for x_name in [
                    "intensity",
                    "return_num",
                    "num_returns",
                    "red",
                    "green",
                    "blue",
                ]
            ],
            dtype=np.float32,
        ).transpose()

        return_num_idx = self.x_features_names.index("return_num")
        occluded_points = x[:, return_num_idx] > 1

        x[:, return_num_idx] = (x[:, return_num_idx]) / (
            self.return_num_normalization_max_value
        )
        num_return_idx = self.x_features_names.index("num_returns")
        x[:, num_return_idx] = (x[:, num_return_idx]) / (
            self.return_num_normalization_max_value
        )

        for idx, c in enumerate(self.x_features_names):
            if c in ["red", "green", "blue"]:
                assert x[:, idx].max() <= self.colors_normalization_max_value
                x[:, idx] = x[:, idx] / self.colors_normalization_max_value
                x[occluded_points, idx] = 0

        rgb_avg = (
            np.asarray(
                [las[x_name] for x_name in ["red", "green", "blue"]], dtype=np.float32
            )
            .transpose()
            .mean(axis=1, keepdims=True)
        )

        x = np.concatenate([x, rgb_avg], axis=1)

        try:
            # for LAS format V1.2
            y = las.classification.array.astype(int)
        except:
            # for  LAS format V1.4
            y = las.classification.astype(int)

        return Data(
            pos=pos,
            x=x,
            y=y,
            las_filepath=las_filepath,
            x_features_names=self.x_features_names,
        )


def _get_data_preparation_parser():
    """Gets a parser with parameters for dataset preparation.

    Returns:
        argparse.ArgumentParser: the parser.
    """
    parser = argparse.ArgumentParser(
        description="Prepare a Lidar dataset for deep learning."
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        default="./split.csv",
        help="Path to csv with a basename (e.g. '123_456.las') and split (train/val/test) columns specifying the dataset split.",
    )
    parser.add_argument(
        "--input_data_dir",
        type=str,
        default="./data/raw/",
        help="Path to folder with las files stored in train/val/test subfolders.",
    )
    parser.add_argument(
        "--prepared_data_dir",
        type=str,
        default="./prepared/",
        help="Path to folder to save Data object train/val/test subfolders.",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default="FR",
    )

    return parser


def main():
    """Code to execute to prepare a new set of Lidar tiles for training 3D segmentaiton neural networks.

    From a data directory containing point cloud in LAS format, and a scv specifying the dataset
    train/val/test split for each file (columns: split, basename, example: "val","123_456.las"),
    split the dataset, chunk the point cloud tiles into smaller subtiles, and prepare each sample
    as a pytorch geometric Data object.

    Echo numbers and colors are scaled to be in 0-1 range. Intensity and average color
    are not scaled b/c they are expected to be standardized later.

    To show help relative to data preparation, run:

        cd lidar_multiclass/datamodules/

        conda activate lidar_multiclass

        python loading.py -h


    """

    parser = _get_data_preparation_parser()
    args = parser.parse_args()
    if args.origin == "FR":
        data_prepper = FrenchLidarDataLogic(**args.__dict__)
    elif args.origin == "CH":
        data_prepper = SwissTopoLidarDataLogic(**args.__dict__)
    else:
        raise KeyError(
            f"Data origin must be either FR (French) or CH (Swiss) (currently: {args.origin})"
        )

    data_prepper.prepare()


if __name__ == "__main__":
    main()
