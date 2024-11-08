## Setup
- Create .comet.config file in base repository dir and add credentials
- Run `make build`

## Run
- Check Makefile for specific commands. Commands will differ depending on the machine this will be running on

### Train
python train2.py --name pylon_complex --data /home/data/bboxes/pylon_complex/yolo/data.yaml --batch 24 --worker 8 --epoch 100 --num_gpus 8 --no_cache_indexing

### Parameters
- H100 Gpu: 
    - batch size: 24
    - workers: 8
    - no_cache_indexing argument

## Notes
- In docker container, you can check super-gradients version by running: `python -c 'import super_gradients; print(super_gradients.__version__)'`