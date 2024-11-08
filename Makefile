IMAGE_NAME?=yolo-nas

export IMAGE_NAME

##### Build Docker Image
build:
	docker build -t $(IMAGE_NAME) . 

##### Run a container by image name
run-fsai:
	docker run --gpus all -it --shm-size=10gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	-v ${PWD}/.comet.config:/root/.comet.config \
	-v ${PWD}/__init__.py:/home/super-gradients/src/super_gradients/common/sg_loggers/__init__.py  \
	-v ${PWD}/cometml_sg_logger.py:/home/super-gradients/src/super_gradients/common/sg_loggers/cometml_sg_logger.py  \
	-v ${PWD}/checkpoint_utils.py:/home/super-gradients/src/super_gradients/training/utils/checkpoint_utils.py  \
	-v ${PWD}/distributed_training_utils.py:/home/super-gradients/src/super_gradients/training/utils/distributed_training_utils.py  \
	-v ${PWD}/train.py:/home/super-gradients/train.py \
	-v ${PWD}/inference.py:/home/super-gradients/inference.py \
	-v /home/fsai/data:/home/data \
	-v /home/fsai/model-output:/home/model-output \
	$(IMAGE_NAME)

run-h100:
	docker run --gpus all -it --shm-size=10gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	-v ${PWD}/.comet.config:/root/.comet.config \
	-v ${PWD}/__init__.py:/home/super-gradients/src/super_gradients/common/sg_loggers/__init__.py  \
	-v ${PWD}/cometml_sg_logger.py:/home/super-gradients/src/super_gradients/common/sg_loggers/cometml_sg_logger.py  \
	-v ${PWD}/checkpoint_utils.py:/home/super-gradients/src/super_gradients/training/utils/checkpoint_utils.py  \
	-v ${PWD}/distributed_training_utils.py:/home/super-gradients/src/super_gradients/training/utils/distributed_training_utils.py  \
	-v ${PWD}/train.py:/home/super-gradients/train.py \
	-v ${PWD}/inference.py:/home/super-gradients/inference.py \
	-v /home/ubuntu/fsai/data:/home/data \
	-v /home/ubuntu/fsai/model-output:/home/model-output \
	$(IMAGE_NAME)
