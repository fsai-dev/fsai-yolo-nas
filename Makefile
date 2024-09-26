IMAGE_NAME?=yolo-nas

export IMAGE_NAME

##### Build Docker Image
build:
	docker build -t $(IMAGE_NAME) . 

##### Run a container by image name
run:
	docker run --gpus all -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 -it --shm-size=10gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v ${PWD}/clearml.conf:/root/clearml.conf -v /home/fsai/data:/home/data -v /home/fsai/model-output/yolo-nas-output:/home/model-output/yolo-nas-output  -v ${PWD}/train.py:/home/super-gradients/train.py -v ${PWD}/inference.py:/home/super-gradients/inference.py $(IMAGE_NAME)
