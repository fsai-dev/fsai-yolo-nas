import os

from typing import Union, Any

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import comet_ml


from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_sg_logger
from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.environment.ddp_utils import multi_process_safe
from super_gradients.common.sg_loggers.time_units import TimeUnit

logger = get_logger(__name__)


def log_config(cfg, experiment):
    """Traverse the Detectron Config graph and log the parameters

    Args:
        cfg (CfgNode): Detectron Config Node
        experiment (comet_ml.Experiment): Comet ML Experiment object
    """

    def log_node(node, prefix):
        if isinstance(node, dict):
            experiment.log_parameters(node, prefix=prefix)
        else:
            experiment.log_parameter(name=prefix, value=node)
            return

        node_dict = dict(node)
        for k, v in node_dict.items():
            _prefix = f"{prefix}-{k}" if prefix else k
            log_node(v, _prefix)

    log_node(cfg, "")


@register_sg_logger("cometml_sg_logger")
class CometmlSGLogger(BaseSGLogger):
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
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

        self.setup(project_name, experiment_name)

        self.save_checkpoints = save_checkpoints_remote
        self.save_tensorboard = save_tensorboard_remote
        self.save_logs = save_logs_remote

    @multi_process_safe
    def setup(self, project_name, experiment_name):
        from multiprocessing.process import BaseProcess

        # Prevent clearml modifying os.fork and BaseProcess.run, which can cause a DataLoader to crash (if num_worker > 0)
        # Issue opened here: https://github.com/allegroai/clearml/issues/790
        default_fork, default_run = os.fork, BaseProcess.run
        self.experiment = comet_ml.Experiment(
            api_key="2ce76zHmN70qw0PxxiQlywYWu",
            project_name="yolo-nas",
            workspace="fsai",
        )

    @multi_process_safe
    def add_config(self, tag: str, config: dict):
        super(CometmlSGLogger, self).add_config(tag=tag, config=config)

        def log_node(node, prefix):
            if isinstance(node, dict):
                self.experiment.log_parameters(node, prefix=prefix)
            else:
                self.experiment.log_parameter(name=prefix, value=node)
                return

        for k, v in config.items():
            log_node(v, k)


    def __add_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.array, Image.Image],
        global_step: int,
    ):
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if image.shape[0] < 5:
            image = image.transpose([1, 2, 0])
        self.clearml_logger.report_image(
            title=tag,
            series=tag,
            image=image,
            iteration=global_step,
            max_image_history=-1,
        )

    @multi_process_safe
    def add_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.array, Image.Image],
        data_format="CHW",
        global_step: int = 0,
    ):
        super(ClearMLSGLogger, self).add_image(
            tag=tag, image=image, data_format=data_format, global_step=global_step
        )
        self.__add_image(tag, image, global_step)

    @multi_process_safe
    def add_images(
        self,
        tag: str,
        images: Union[torch.Tensor, np.array],
        data_format="NCHW",
        global_step: int = 0,
    ):
        super(ClearMLSGLogger, self).add_images(
            tag=tag, images=images, data_format=data_format, global_step=global_step
        )
        for image in images:
            self.__add_image(tag, image, global_step)


    @multi_process_safe
    def close(self):
        super().close()
        self.task.close()

    @multi_process_safe
    def add_file(self, file_name: str = None):
        super().add_file(file_name)
        self.task.upload_artifact(
            name=file_name, artifact_object=os.path.join(self._local_dir, file_name)
        )

    @multi_process_safe
    def upload(self):
        super().upload()

        if self.save_tensorboard:
            name = self._get_tensorboard_file_name().split("/")[-1]
            self.task.upload_artifact(
                name=name, artifact_object=self._get_tensorboard_file_name()
            )

        if self.save_logs:
            name = self.experiment_log_path.split("/")[-1]
            self.task.upload_artifact(
                name=name, artifact_object=self.experiment_log_path
            )

    @multi_process_safe
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        state_dict = self._sanitize_checkpoint(state_dict)

        name = f"ckpt_{global_step}.pth" if tag is None else tag
        if not name.endswith(".pth"):
            name += ".pth"

        path = os.path.join(self._local_dir, name)
        torch.save(state_dict, path)

        if self.save_checkpoints:
            if self.s3_location_available:
                self.model_checkpoints_data_interface.save_remote_checkpoints_file(
                    self.experiment_name, self._local_dir, name
                )
            self.task.upload_artifact(name=name, artifact_object=path)

    def add(self, tag: str, obj: Any, global_step: int = None):
        pass
