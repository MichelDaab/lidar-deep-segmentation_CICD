import os
import pickle
import hydra
import laspy
import torch
from omegaconf import DictConfig
from typing import Optional
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
)
from tqdm import tqdm

from semantic_val.utils.db_communication import ConnectionData
from semantic_val.decision.codes import reset_classification
from semantic_val.utils import utils
from semantic_val.datamodules.processing import DataHandler

from semantic_val.decision.decide import (
    prepare_las_for_decision,
    update_las_with_decisions,
)


log = utils.get_logger(__name__)


@utils.eval_time
def predict(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Those are the 4 needed inputs
    assert os.path.exists(config.prediction.resume_from_checkpoint)
    assert os.path.exists(config.prediction.src_las)
    assert os.path.exists(config.prediction.best_trial_pickle_path)


    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_all_transforms()
    datamodule._set_predict_data(
        [config.prediction.src_las], config.prediction.mts_auto_detected_code
    )

    data_handler = DataHandler(preds_dirpath=config.prediction.output_dir)
    data_handler.load_las_for_proba_update(config.prediction.src_las)

    with torch.no_grad():
        device =  torch.device('cuda') if "gpus" in config.trainer and config.trainer.gpus == 1 else torch.device('cpu')

        model: LightningModule = hydra.utils.instantiate(config.model) 
        model = model.load_from_checkpoint(config.prediction.resume_from_checkpoint)
        model.to(device)
        model.eval()

        for index, batch in tqdm(
            enumerate(datamodule.predict_dataloader()), desc="Batch inference..."
        ):
            batch.to(device)
            outputs = model.predict_step(batch)
            data_handler.append_pos_and_proba_to_list(outputs)
            # if index > 2:
            #    break  ###### à supprimer ###################

    updated_las_path = data_handler.interpolate_probas_and_save("predict")

    data_connexion_db = ConnectionData(config.prediction.host, config.prediction.user, config.prediction.pwd, config.prediction.bd_name)

    log.info("Prepare LAS...")
    prepare_las_for_decision(
        updated_las_path,
        data_connexion_db,
        updated_las_path,
        candidate_building_points_classification_code=[
            config.prediction.mts_auto_detected_code
        ],
    )

    log.info("Update classification...")
    las = laspy.read(updated_las_path)
    with open(config.prediction.best_trial_pickle_path, "rb") as f:
        log.info(f"Using best trial from: {config.prediction.best_trial_pickle_path}")
        best_trial = pickle.load(f)

    use_final_classification_codes = True if "use_final_classification_codes" in\
         config.prediction and config.prediction.use_final_classification_codes == True else False

    las = update_las_with_decisions(
        las, 
        best_trial.params, 
        use_final_classification_codes, 
        config.prediction.mts_auto_detected_code
    )
    las.write(updated_las_path)
    log.info(f"Updated LAS saved to : {updated_las_path}")
