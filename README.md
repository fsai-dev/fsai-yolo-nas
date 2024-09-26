## Setup
- Create clearml.conf file in base repo dir and add credentials
- Run `make build`


## Run
- Run `make run`

### Train
python train.py --name pylon_complex --data /home/data/bboxes/pylon_complex/data.yaml --batch 8 --worker 10 --epoch 300 --model yolo_nas_l --size 1024 --num_gpus 4

### Inference
python inference.py --project_name 300_epoch --num_classes 1 --conf 0.5 --source /home/data/bboxes/pylon_complex/val.json --weight /home/yolo-nas-output/runs/pylon_complex/RUN_20240827_232659_396567/ckpt_latest.pth --save --hide

# 300 epoch run: RUN_20240827_232659_396567
# 100 epoch run: RUN_20240824_001613_199026




## Notes
- In docker container, you can check super-gradients version by running: `python -c 'import super_gradients; print(super_gradients.__version__)'`