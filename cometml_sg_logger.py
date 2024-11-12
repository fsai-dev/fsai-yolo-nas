import os

import torch
from typing import Union, Any
import numpy as np
from PIL import Image


from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_sg_logger
from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.environment.ddp_utils import multi_process_safe
from super_gradients.common.sg_loggers.time_units import TimeUnit

logger = get_logger(__name__)

try:
    import comet_ml

    _import_comet_ml_error = None
except (ModuleNotFoundError, ImportError, NameError) as cometml_import_err:
    logger.debug("Failed to import comet_ml")
    _import_comet_ml_error = cometml_import_err


@register_sg_logger("cometml_sg_logger")
class CometMLSGLogger(BaseSGLogger):
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        base_data_dir: str,
        storage_location: str,
        resumed: bool,
        training_params: dict,
        checkpoints_dir_path: str,
        tb_files_user_prompt: bool = False,
        launch_tensorboard: bool = False,
        tensorboard_port: int = None,
        save_checkpoints_remote: bool = True,
        save_tensorboard_remote: bool = False,
        save_logs_remote: bool = True,
        monitor_system: bool = None,
    ):
        """
        :param project_name:            ClearML project name that can include many experiments
        :param experiment_name:         Name used for logging and loading purposes
        :param storage_location:        If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param resumed:                 If true, then old tensorboard files will **NOT** be deleted when tb_files_user_prompt=True
        :param training_params:         training_params for the experiment.
        :param checkpoints_dir_path:    Local root directory path where all experiment logging directories will reside.
        :param tb_files_user_prompt:    Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard:      Whether to launch a TensorBoard process.
        :param tensorboard_port:        Specific port number for the tensorboard to use when launched (when set to None, some free port number will be used
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote:        Saves log files in s3.
        :param monitor_system:          Not Available for ClearML logger. Save the system statistics (GPU utilization, CPU, ...) in the tensorboard
        """

        self.s3_location_available = storage_location.startswith("s3")
        super().__init__(
            project_name=project_name,
            experiment_name=experiment_name,
            storage_location=storage_location,
            resumed=resumed,
            training_params=training_params,
            checkpoints_dir_path=checkpoints_dir_path,
            tb_files_user_prompt=tb_files_user_prompt,
            launch_tensorboard=launch_tensorboard,
            tensorboard_port=tensorboard_port,
            save_checkpoints_remote=self.s3_location_available,
            save_tensorboard_remote=self.s3_location_available,
            save_logs_remote=self.s3_location_available,
            monitor_system=False,
        )
        self.checkpoints_dir_path = self._local_dir
        self.base_data_dir = base_data_dir
        self.save_checkpoints = save_checkpoints_remote
        self.save_tensorboard = save_tensorboard_remote
        self.save_logs = save_logs_remote

        if _import_comet_ml_error:
            raise _import_comet_ml_error

        comet_ml.login()
        self.setup(project_name, experiment_name)

    @multi_process_safe
    def setup(self, project_name, experiment_name):
        self.experiment = comet_ml.Experiment(project_name=project_name)
        self.experiment.set_name(experiment_name)
        self.experiment.log_asset(
            file_data=os.path.join(self.base_data_dir, "annotation_stats.json")
        )
        self.experiment.log_asset(
            file_data=os.path.join(self.base_data_dir, "val.json")
        )
        self.experiment.log_asset(
            file_data=os.path.join(self.base_data_dir, "train.json")
        )
        self.experiment.log_asset(
            file_data=os.path.join(self.base_data_dir, "class_names.yaml")
        )

    @multi_process_safe
    def add_config(self, tag: str, config: dict):
        config["training_hyperparams"]["sg_logger_params"][
            "checkpoints_dir_path"
        ] = self._local_dir
        super(CometMLSGLogger, self).add_config(tag=tag, config=config)

        def log_node(node, prefix):
            if isinstance(node, dict):
                self.experiment.log_parameters(node, prefix=prefix)
            else:
                self.experiment.log_parameter(name=prefix, value=node)
                return

        for k, v in config.items():
            log_node(v, k)

    @multi_process_safe
    def add_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.array, Image.Image],
        data_format="CHW",
        global_step: int = 0,
    ):
        super(CometMLSGLogger, self).add_image(
            tag=tag, image=image, data_format=data_format, global_step=global_step
        )
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if image.shape[0] < 5:
            image = image.transpose([1, 2, 0])
        self.experiment.log_image(image_data=image, name=tag, step=global_step)

    @multi_process_safe
    def add_images(
        self,
        tag: str,
        images: Union[torch.Tensor, np.array],
        data_format="NCHW",
        global_step: int = 0,
    ):
        super(CometMLSGLogger, self).add_images(
            tag=tag, images=images, data_format=data_format, global_step=global_step
        )
        for image in images:
            self.add_image(tag=tag, image=image, global_step=global_step)

    @multi_process_safe
    def add_scalar(
        self, tag: str, scalar_value: float, global_step: Union[int, TimeUnit] = 0
    ):
        super(CometMLSGLogger, self).add_scalar(
            tag=tag, scalar_value=scalar_value, global_step=global_step
        )
        if isinstance(global_step, TimeUnit):
            global_step = global_step.get_value()
        self.experiment.log_metric(
            name=tag, value=scalar_value, step=global_step, epoch=global_step
        )

    @multi_process_safe
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        super(CometMLSGLogger, self).add_scalars(
            tag_scalar_dict=tag_scalar_dict, global_step=global_step
        )
        self.experiment.log_metrics(
            dic=tag_scalar_dict, step=global_step, epoch=global_step
        )

    @multi_process_safe
    def close(self):
        super().close()
        self.experiment.end()

    @multi_process_safe
    def add_file(self, file_name: str = None):
        super().add_file(file_name)
        self.experiment.log_asset(
            file_data=file_name, file_name=file_name, overwrite=True
        )

    @multi_process_safe
    def upload(self):
        super().upload()
        self.experiment.log_asset(file_data=self.experiment_log_path, overwrite=True)

    @multi_process_safe
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        state_dict = self._sanitize_checkpoint(state_dict)

        name = f"ckpt_{global_step}.pth" if tag is None else tag
        if not name.endswith(".pth"):
            name += ".pth"

        path = os.path.join(self._local_dir, name)
        torch.save(state_dict, path)

        if self.save_checkpoints:
            self.experiment.log_model(
                name="yolonas",
                file_or_folder=path,
                file_name=name,
                overwrite=True,
            )

    def add(self, tag: str, obj: Any, global_step: int = None):
        pass
