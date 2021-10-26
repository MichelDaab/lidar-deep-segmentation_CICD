import os
from typing import List, Optional

from omegaconf import DictConfig
from pytorch_lightning import seed_everything
import glob
import os.path as osp
from tqdm import tqdm
from semantic_val.utils import utils
from semantic_val.validation.validation_utils import (
    ShapeFileCols,
    compare_classification_with_predictions,
    make_decisions,
    vectorize_into_candidate_building_shapes,
    load_geodf_of_candidate_building_points,
)

log = utils.get_logger(__name__)


# TODO: Use append mode in to_file mode=”a” to avoid ugly pandas concatenation
# TODO: Become independant of post-correction shapefile by creating shapefil "as if" it did not undergo correction ?
# We have Classification in the LAS files so we can evaluate method on the newly created shapefile.
# (we can identify surclassification on the fly with associated points).


def validate(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    log.info(f"Logging directory: {os.getcwd()}")
    las_filepath = glob.glob(
        osp.join(config.validation_module.predicted_las_dirpath, "*.las")
    )
    for las_filepath in tqdm(las_filepath, desc="Evaluating predicted point cloud"):
        log.info(f"Evaluation of tile {las_filepath}...")

        las_gdf = load_geodf_of_candidate_building_points(las_filepath)
        if len(las_gdf) == 0:
            log.info("/!\ Skipping tile with no candidate building points.")
            continue
        shapes_gdf = vectorize_into_candidate_building_shapes(las_gdf)
        comparison = compare_classification_with_predictions(shapes_gdf, las_gdf)
        comparison = make_decisions(comparison)

        df_out = shapes_gdf.join(comparison, on="shape_idx", how="left")
        keep = [item.value for item in ShapeFileCols] + ["geometry"]
        df_out = df_out[keep]

        output_shp = osp.join(
            os.getcwd(), config.validation_module.operationnal_output_shapefile_name
        )
        mode = "w" if not osp.isfile(output_shp) else "a"
        df_out.to_file(output_shp, mode=mode, index=False)
    log.info(f"Output shapefile is in {output_shp}")
