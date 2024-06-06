## HF2-VAD-AD
Official implementation of "Hybrid Video Anomaly Detection for Anomalous Scenarios in Autonomous Driving".

![pipeline](./assets/Paper_Ablauf.png)
## 1. Dependencies
```
python==3.6
pytorch==1.5.1
mmcv-full==1.3.1
mmdet==2.11.0
scikit-learn==0.23.2
edflow==0.4.0
PyYAML==5.4.1
tensorboardX==2.4
```
## 2. Usage
### 2.1 Data preparation
Please follow the [instructions](./pre_process/readme.md) to prepare the training and testing dataset.

### 2.2 Train
We train the ML-MemAE-SC at first, then train CVAE model with the reconstructed flows,
and finally finetune the whole framework. All the config files are located at `./cfgs`. 

To train the ML-MemAE-SC, run:
```python
$ python ml_memAE_sc_train.py
```
To train the CVAE model with reconstructed flows, run:
```python
$ python train.py
```
And finetune the whole HF2VAD-AD framework together as:
```python
$ python finetune.py
```
For different datasets, please modify the configuration files accordingly.

### 2.3 Evaluation
To evaluation the anomaly detection performance of the trained model, run:
```python
$ python eval.py [--model_save_path] [--cfg_file] 
```
E.g., for the Carla dataset:
```python
$ python eval.py \
         --model_save_path=./pretrained_ckpts/carla_local_best.pth \
         --cfg_file=./pretrained_ckpts/carla_local_best.yaml
```
You can download the pretrained weights of HF2VAD-AD for Carla_Local datasets 
from [here](https://drive.google.com/file/d/1q9yXnHrMNK8qut8nn_mWiVd1HGBoGd-F/view?usp=sharing), the training stats from [here](https://drive.google.com/file/d/1JuxBioxPyhB5b2g82IwQ77WbY22lR0GZ/view?usp=drive_link)  the config from [here](https://drive.google.com/file/d/1tGI5_Us5Wr24K5-TSmSMykV3BOjgMlSU/view?usp=sharing)

## Acknowledgment
We thank LiUzHiAn for the PyTorch implementation of the [HF2VAD](https://github.com/LiUzHiAn/hf2vad).

## Citation

```

```
