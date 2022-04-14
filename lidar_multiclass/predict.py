import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from tqdm import tqdm

from lidar_multiclass.utils import utils
from lidar_multiclass.models.interpolation import Interpolator


log = utils.get_logger(__name__)
torch.set_grad_enabled(False)


def predict(config: DictConfig) -> str:
    """
    Inference pipeline.

    A lightning datamodule splits a single point cloud of arbitrary size (typically: 1km * 1km) into subtiles
    (typically 50m * 50m), which are grouped into batches that are fed to a trained neural network embedded into a lightning Module.

    Predictions happen on a subsampled version of each subtile, which needs to be propagated back to the complete
    point cloud via an Interpolator. This Interpolator also includes the creation of a new LAS file with additional
    dimensions, including predicted classification, entropy, and (optionnaly) predicted probability for each class.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        str: path to ouptut LAS.

    """

    # Those are the 2 needed inputs, in addition to the hydra config.
    assert os.path.exists(config.predict.resume_from_checkpoint)
    assert os.path.exists(config.predict.src_las)

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_predict_data([config.predict.src_las])

    model: LightningModule = hydra.utils.instantiate(config.model)
    model = model.load_from_checkpoint(config.predict.resume_from_checkpoint)
    device = utils.define_device_from_config_param(config.predict.gpus)
    model.to(device)
    model.eval()

    itp = Interpolator(
        output_dir=config.predict.output_dir,
        classification_dict=datamodule.dataset_description.get("classification_dict"),
        probas_to_save=config.predict.probas_to_save,
    )

    for batch in tqdm(datamodule.predict_dataloader()):
        batch.to(device)
        outputs = model.predict_step(batch)
        itp.update(outputs)

    out_f = itp.interpolate_and_save()
    return out_f


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):
    """See function {predict.__name__}.

    :meta private:

    """
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from lidar_multiclass.utils import utils
    from lidar_multiclass.predict import predict

    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    if config.get("print_config"):
        utils.print_config(config, resolve=False)

    return predict(config)


if __name__ == "__main__":
    # cf. https://github.com/facebookresearch/hydra/issues/1283
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    main()
