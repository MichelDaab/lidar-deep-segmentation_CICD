"""Utilities to analyse point cloud data post-prediction"""

import glob
import os
import sys
from typing import List, Tuple
import numpy as np

import pdal


rel_root_path = "./"  # we run from project root and add it to the path
abs_root_path = os.path.abspath(rel_root_path)
print(abs_root_path)
sys.path.insert(0, abs_root_path)

import json
import subprocess
from tempfile import NamedTemporaryFile
from lidar_multiclass.utils import utils


def run_pdal_info(las: utils.PathLike, out_metadata_file: utils.PathLike) -> None:
    """Runs PDAL info, which is the faster way to get metadata from a point cloud.

    Args:
        las (PathLike): path to input
        out_metadata_file (PathLike): path to saved metadata file
    """
    command = f"pdal info {las} --metadata > {out_metadata_file}"
    print(command)
    subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        check=True,
    )


def calc_entropy_stats(
    las: utils.PathLike, output_dir: utils.PathLike
) -> Tuple[float, np.array]:
    """Returns average entropy and deciles of entropy"""

    pipeline = pdal.Pipeline() | pdal.Reader.las(filename=las)
    pipeline.execute()
    data = pipeline.arrays[0]
    e = data["entropy"]
    deciles_q = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    avg = np.average(e)
    deciles = np.quantile(e, q=deciles_q)
    data["entropy"] = avg

    output_tif = os.path.join(output_dir, os.path.basename(las).replace(".las", ".tif"))

    pipeline = pdal.Writer.gdal(
        filename=output_tif,
        dimension="entropy",
        data_type="float",
        output_type="mean",
        window_size=0.5,
        resolution=1,
        nodata=0,
    ).pipeline(data)
    pipeline.execute()
    return avg, deciles


def summarize_entropies(
    files: List[utils.PathLike], output_dir: utils.PathLike
) -> None:

    for las in files:
        avg, deciles = calc_entropy_stats(las, output_dir)
        print(las)
        print(f"average: {avg}, deciles: {deciles}")


# For now this analysis is hardoced. After delivery, clean and document everything in a specific script.
files = glob.glob("/var/data/cgaydon/data/temp/batch_6_predicted/89300*.las")
output_dir = "/home/cgaydon/repositories/custom_analyses/20220406_entropy_maps/outputs/"
summarize_entropies(files, output_dir)
