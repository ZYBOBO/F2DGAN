# F2DGAN-pytorch

The demo implementation of our paper F2DGAN: Exact Fusion via Feature Distribution Matching for Few-shot Image Generation.

## Prerequisites
- Python 3.8
- Pytorch 1.8
- Nvidia GPU + CUDA

## Preparing Dataset
Download the [datasets](https://portland-my.sharepoint.com/:f:/g/personal/zhenggu4-c_my_cityu_edu_hk/ErQRAfnkT1xJqaTZwB7ZVWoBrAu86flhwQeuBoHMS-bfVA?e=gaaeAZ) and unzip them in `datasets` folder.

## Training
```shell
python train.py --conf configs/flower_f2dgan.yaml \
--output_dir results/flower_f2dgan \
--gpu 0
```

* You may also customize the parameters in `configs`.


## Testing
```shell
python test.py --name results/flower_f2dgan --gpu 0
```

The generated images will be saved in `results/flower_f2dgan/test`.


## FID and LPIPS Evaluation
```shell
python main_metric.py --gpu 0 --dataset flower \
--name results/flower_f2dgan \
--real_dir datasets/for_fid/flower --ckpt gen_00100000.pt \
--fake_dir test_for_fid
```
