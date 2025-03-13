# Simplifying DINO via Coding Rate Regularization

PyTorch implementation and pretrained models for SimDINO and SimDINOv2.

<div align="center">
  <image src="assets/pipelines.png" width="840px" />
  <p></p>
</div>

Authors: [Ziyang Wu](https://robinwu218.github.io/), [Jingyuan Zhang](https://github.com/ultranity), [Druv Pai](https://druvpai.github.io/), [Xudong Wang](https://people.eecs.berkeley.edu/~xdwang/), [Chandan Singh](https://csinva.io/), [Jianwei Yang](https://jwyang.github.io/), [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/), [Yi Ma](https://people.eecs.berkeley.edu/~yima/)

[[`Paper`](https://arxiv.org/abs/2502.10385)] [[`Website`](https://robinwu218.github.io/SimDINO)] [[`BibTeX`](#citing-simdino)]

## Update
[02/25/25] We release code and pretrained checkpoints for SimDINO and SimDINOv2.

## Pretrained models
We provide checkpoints for both SimDINO and SimDINOv2 pretrained on ImageNet-1k for 100 epochs following configs detailed in our [paper](https://arxiv.org/abs/2502.10385).

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th>Algorithm</th>
      <th>ImageNet<br />k-NN</th>
      <th>ImageNet<br />linear</th>
      <th>download</th>
    </tr>
  </thead>
  <tbody>
  <tr>
      <td>ViT-B/16</td>
      <td align="right">86 M</td>
      <td align="center">SimDINO</td>
      <td align="right">74.9%</td>  
      <td align="right">77.3%</td>
      <td><a href="https://drive.google.com/file/d/1nqcX0IgKQ8H3ZFJKaEx_O5wTVqdorLyQ/view?usp=drive_link">ckpt</a></td>
    </tr>
    <tr>
      <td>ViT-L/16</td>
      <td align="right">300 M</td>
      <td align="center">SimDINO</td>
      <td align="right">75.6%</td>
      <td align="right">77.4%</td>
      <td><a href="https://drive.google.com/file/d/1jy27VYHHSxllMqWr5DiKk80y9ZBrKkPV/view?usp=drive_link">ckpt</a></td>
    </tr>
    <tr>
      <td>ViT-B/16</td>
      <td align="right">86 M</td>
      <td align="center">SimDINOv2</td>
      <td align="right">78.1%</td>
      <td align="right">79.7%</td>
      <td><a href="https://drive.google.com/file/d/1g_f3aEFdfiKCn8IH11Y4FwtEDu1pPbQv/view?usp=drive_link">ckpt</a></td>
    </tr>
    <tr>
      <td>ViT-L/16</td>
      <td align="right">300 M</td>
      <td align="center">SimDINOv2</td>
      <td align="right">81.1%</td>
      <td align="right">82.4%</td>
      <td><a href="https://drive.google.com/file/d/1IotEWszh1chGtzwl8X_ul6j3ACT7Tn6i/view?usp=drive_link">ckpt</a></td>
    </tr>
  </tbody>
</table>

Below we also provide the checkpoints for the original DINO and DINOv2 models that we trained.

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th>Algorithm</th>
      <th>ImageNet<br />k-NN</th>
      <th>ImageNet<br />linear</th>
      <th>download</th>
    </tr>
  </thead>
  <tbody>
  <tr>
      <td>ViT-B/16</td>
      <td align="right">86 M</td>
      <td align="center">DINO</td>
      <td align="right">72.9%</td>
      <td align="right">76.3%</td>
      <td><a href="https://drive.google.com/file/d/1o-FtJ0TDNPAZiREa_qjGENksYL_3Ohc8/view?usp=drive_link">ckpt</a></td>
    </tr>
    <tr>
      <td>ViT-B/16</td>
      <td align="right">86 M</td>
      <td align="center">DINOv2</td>
      <td align="right">76.0%</td>
      <td align="right">77.2%</td>
      <td><a href="https://drive.google.com/file/d/1QZpUkC_yX2V9rWqHW3qLTFfvD6BUqyN_/view?usp=drive_link">ckpt</a></td>
    </tr>
    <tr>
      <td>ViT-L/16</td>
      <td align="right">300 M</td>
      <td align="center">DINOv2</td>
      <td align="right">80.8%</td>
      <td align="right">82.0%</td>
      <td><a href="https://drive.google.com/file/d/1p-cUWNShnZEFBJ_mnw77HnvCFIV0FthN/view?usp=drive_link">ckpt</a></td>
    </tr>
  </tbody>
</table>

Note: our compute resource is limited but we are working on scaling up our approach. Stay tuned for more model checkpoints in the future. Meanwhile, we always welcome and appreciate feedback and help from the community. 

## Installation

Our implementation requires Python 3.11+, PyTorch 2.4+ and [xFormers](https://github.com/facebookresearch/xformers) 0.0.29+ and some other packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup the dependencies, please install via:

```sh
pip install -r requirements.txt
```


## Data preparation

First, you need to download the [ImageNet-1k](https://www.image-net.org/download.php) dataset.

The root directory of the dataset should hold the following contents:

- `<ROOT>/test/ILSVRC2012_test_00000001.JPEG`
- `<ROOT>/test/[..]`
- `<ROOT>/test/ILSVRC2012_test_00100000.JPEG`
- `<ROOT>/train/n01440764/n01440764_10026.JPEG`
- `<ROOT>/train/[...]`
- `<ROOT>/train/n15075141/n15075141_9993.JPEG`
- `<ROOT>/val/n01440764/ILSVRC2012_val_00000293.JPEG`
- `<ROOT>/val/[...]`
- `<ROOT>/val/n15075141/ILSVRC2012_val_00049174.JPEG`
- `<ROOT>/labels.txt`

Specific to SimDINOv2, you need to configure and run `python prepare.py` to generate some metadata files.
The generated files should have the following structure:

- `<EXTRA>/class-ids-TRAIN.npy`
- `<EXTRA>/class-ids-VAL.npy`
- `<EXTRA>/class-names-TRAIN.npy`
- `<EXTRA>/class-names-VAL.npy`
- `<EXTRA>/entries-TEST.npy`
- `<EXTRA>/entries-TRAIN.npy`
- `<EXTRA>/entries-VAL.npy`


## Training

### Training SimDINO on ImageNet-1k

You can train SimDINO on ViT-B/16 with an 8-GPU node (each with at least 40G memory):

```shell
cd simdino
torchrun --nnodes=1 --nproc_per_node=8 main_dino.py --arch vit_base --patch_size 16 --local_crops_number 10 \
    --eps 0.05 --coeff 1 --output_dir <PATH/TO/OUTPUT/DIR> --data_path <PATH/TO/DATASET/TRAIN> \
    --track_wandb # to enable logging; use --track_wandb to log with wandb and --track_swan to log with swanlab
```
Training time is approximately 1.5 day and you should be able to replicate our reported results. An example log on ViT-B/16 can be found [here](assets/SimDNIOv1_vitb16.txt).

### Training SimDINOv2 on ImageNet-1k

You can train SimDINOv2 on ViT-L/16 with a 8-GPU node (each with at least 40G memory):

```shell
torchrun --nnodes=1 --nproc_per_node=8 simdinov2/train/train.py \
    --config-file simdinov2/configs/simdino_config.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

Training time is approximately 1 day and you should be able to replicate our reported results. An example log on ViT-B/16 can be found [here](assets/SimDNIOv2-vitb16.json).
The training code saves the weights of the teacher in the `eval` folder every 10 epochs for evaluation. You can change the `student.arch` field in `simdino_config.yaml` to train other models.

You can also use `submitit` if your environment happens to be a SLURM cluster:
```shell
python simdinov2/run/train/train.py \
    --nodes 1 \
    --config-file simdinov2/configs/simdino_config.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### FAQ & Tips on Training
**Q: How can I visualize the training losses?**

**A:** In SimDINO, you can remove the `--nowandb` argument to enable wandb logging.

**Q: I notice some spikes in coding rate loss in early training stages. Is that normal?**

**A:** Occasional spikes are normal and shouldn't impact final performance. If you notice too much instability, the following operations can help:
- set `--expa_type=1`. Sometimes spikes are caused by sudden change in conditioning of the covariance matrix and this applies some "smoothing" by centering the student features and teacher features.
- set a smaller `--eps`. 

**Q: I can only use small batch sizes per gpu for training, what should I do?**

**A:** You can set `--reduce_cov=1` to collect covariance matrices from multiple gpus via all_reduce. Empirically, we found that we don't have to do this even with 64 samples per GPU.


## Evaluation

The teacher weights are regularly saved and can be evaluated using the following scripts.

### k-NN classification on ImageNet-1k with SimDINO
For example, on ViT-B/16:
```shell
cd simdino
torchrun --nproc_per_node=8 eval_knn.py --patch_size 16 --arch vit_base \
    --pretrained_weights <PATH/TO/OUTPUT/DIR>/checkpoint.pth --data_path <PATH/TO/DATASET>
```

### Linear probing on ImageNet-1k with SimDINO
For example, on ViT-B/16:
```shell
cd simdino
torchrun --nproc_per_node=8 eval_linear.py --patch_size 16 --arch vit_base \
    --pretrained_weights <PATH/TO/OUTPUT/DIR>/checkpoint.pth --data_path <PATH/TO/DATASET>
```

### k-NN classification on ImageNet-1k with SimDINOv2

```shell
python simdinov2/run/eval/knn.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/knn \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### Linear probing on ImageNet-1k with SimDINOv2

```shell
python simdinov2/run/eval/linear.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/linear \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```


## Citing SimDINO

If you find this project useful, please consider giving us a star and citation:
```
@article{wu2025simplifying,
  title={Simplifying DINO via Coding Rate Regularization},
  author={Wu, Ziyang and Zhang, Jingyuan and Pai, Druv and Wang, XuDong and Singh, Chandan and Yang, Jianwei and Gao, Jianfeng and Ma, Yi},
  journal={arXiv preprint arXiv:2502.10385},
  year={2025}
}
```

## Acknowledgements

This project is largely built upon the orignal [DINO](https://github.com/facebookresearch/dino) and [DINOv2](https://github.com/facebookresearch/dinov2) projects. 