
## Data preprocessing

### 0. Dataset preparing
Here we use the `carla` dataset as example. For the other datasets (ped2, Avenue, ShanghaiTech) follow the original instructions.

Download the [Temporal Anomaly (Sudden Braking) Dataset](https://zenodo.org/records/12269929) and place it into 
the `data` directory of this project. In order to evaluate the frame-level AUC, we provide the 
frame labels of each test video in `data/carla_local/ground_truth_demo/gt_label.json`. 

The file structure should be similar as follows:
```python
./data
└── carla_local
    ├── ground_truth_demo
    │   └── gt_label.json
    ├── testing
    │   └── frames
    │       ├── 0001
                ├──ground-truth
                ├──RGB_IMG
                ...
            ├── 0002
                ├──ground-truth
                ├──RGB_IMG
            ...
            
```

### 1. Objects detecting

Run the following command to detect all the foreground objects (For the other datasets follow the original instructions). 
```python
$ python extract_bboxes.py [--proj_root] [--dataset_name] [--mode] 
```
E.g., to extract objects of all training data:
```python
$ python extract_bboxes.py --proj_root=<path/to/project_root> --dataset_name=carla_local --mode=train
```
To extract objects of all test data:
```python
$ python extract_bboxes.py --proj_root=<path/to/project_root> --dataset_name=carla_local --mode=test
```

After doing this, the results will be default saved at `./data/carla_local/carla_local_bboxes_train.npy`, 
in which each item contains all the bounding boxes in a single video frame.

### 2. Extracting optical flows
We extract optical flows in videos using use [FlowNet2.0](https://github.com/NVIDIA/flownet2-pytorch). 

1. download the pre-trained FlowNet2 weights (i.e., `FlowNet2_checkpoint.pth.tar`) from [here](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing) 
and place it in `pre_process/assets`.
2. build the customer layers via executing `install_custome_layers.sh`.
3. run the following command to estimate all the optical flows:
```python
$ python extract_flows.py [--proj_root] [--dataset_name] [--mode] 
```
E.g., to extract flows of all training data:
```python
$ python extract_flows.py --proj_root=<path/to/project_root> --dataset_name=carla_local --mode=train
```
To extract flows of all test data:
```python
$ python extract_flows.py --proj_root=<path/to/project_root> --dataset_name=carla_local --mode=test
```

After doing this, the estimated flows will be default saved at `./data/carla_local/training/flows`.
The final data structure should be similar as follows:
```python
```
### 3. Prefetch spatial-temporal cubes
For every extracted object above, we can construct a spatial-temporal cube (STC). These STCs can be also downloaded from [here](https://zenodo.org/records/12269929/)
For example, assume we extract only one bbox in $i$-th frame, then we can crop the same region
from $(i-4), (i-3), (i-2), (i-1), i$ frames using the coordinates of that bbox, resulting a STC 
with shape `[5,3,H,W]`. Things are similar for the optical flows.

To extract all the STCs in the dataset, run the following command:
```python
$ python extract_samples.py [--proj_root] [--dataset_name] [--mode] 
```
E.g., to extract samples of all training data:
```python
$ python extract_samples.py --proj_root=<path/to/project_root> --dataset_name=carla_local --mode=train
```
To extract samples of all test data:
```python
$ python extract_samples.py --proj_root=<path/to/project_root> --dataset_name=carla_local --mode=test
```
Note that the extracted samples number will be very large for Avenue and ShanghaiTech dataset,
hence we save the samples in a chunked file manner. The max number of samples in a separate
chunked file is set to be `100K` by default, feel free to modify that in [#Line11 here](./extract_samples.py)
depending on the available memory and disk space of your machine.

Given the first 4 frames and corresponding flows as input, the model is encouraged to predict the final frame.

```
