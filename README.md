# Bi-FRN

Code relaese for [Bi-Directional Feature Reconstruction Network for Fine-grained Few-shot Image Classification](https://arxiv.org/abs/2211.17161). (Accepted in AAAI-23)

## Code environment

* You can create a conda environment with the correct dependencies using the following command lines:

  ```shell
  conda env create -f environment.yml
  conda activate BiFRN
  ```

## Dataset

The official link of CUB-200-2011 is [here](http://www.vision.caltech.edu/datasets/cub_200_2011/). The preprocessing of the cropped CUB-200-2011 is the same as [FRN](https://github.com/Tsingularity/FRN), but the categories  of train, val, and test follows split.txt. And then move the processed dataset  to directory ./data.

- CUB_200_2011 \[[Download Link](https://drive.google.com/file/d/1WxDB3g3U_SrF2sv-DmFYl8LS0p_wAowh/view)\]
- cars \[[Download Link](https://drive.google.com/file/d/1ImEPQH5gHpSE_Mlq8bRvxxcUXOwdHIeF/view?usp=drive_link)\]
- dogs \[[Download Link](https://drive.google.com/file/d/13avzK22oatJmtuyK0LlShWli00NsF6N0/view?usp=drive_link)\]

## Train

* To train Bi-FRN on `CUB_fewshot_cropped` with Conv-4 backbone under the 1/5-shot setting, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/BiFRN/Conv-4
  ./train.sh
  ```

* For ResNet-12 backbone, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/BiFRN/ResNet-12
  ./train.sh
  ```

## Test

```shell
    cd experiments/CUB_fewshot_cropped/BiFRN/Conv-4
    python ./test.py

    cd experiments/CUB_fewshot_cropped/BiFRN/ResNet-12
    python ./test.py
```

## Traffic sign anomaly classification

The repository now includes end-to-end scripts for training and deploying a binary
classifier on a traffic-sign dataset with a directory layout of:

```
dataset_root/
├── train
│   ├── abnormal
│   └── normal
└── val
    ├── abnormal
    └── normal
```

To train a model that handles the heavy class imbalance, run:

```shell
python experiments/traffic_sign/train_binary.py \
  --data-root /path/to/dataset_root \
  --output-dir outputs/traffic_sign \
  --epochs 60 \
  --batch-size 32
```

Alternatively, in keeping with the other experiment folders you can run the
bundled shell script:

```shell
cd experiments/traffic_sign
DATA_ROOT=/path/to/dataset_root ./train.sh
```

You can optionally set `OUTPUT_DIR` to change where checkpoints are written and
append extra flags (for example `--resnet`) after the script invocation.

The training script automatically applies a weighted sampler and class-weighted
loss to upweight the rare abnormal samples. After training you can evaluate the
best checkpoint on the validation set to obtain precision, recall and confusion
matrix statistics via:

```shell
python experiments/traffic_sign/eval_binary.py \
  --data-root /path/to/dataset_root \
  --checkpoint outputs/traffic_sign/best_model.pth
```

Finally, single-image inference can be performed with:

```shell
python experiments/traffic_sign/predict_single.py \
  --data-root /path/to/dataset_root \
  --checkpoint outputs/traffic_sign/best_model.pth \
  --image /path/to/sample.jpg
```

All three scripts accept a `--resnet` flag if you prefer the ResNet-12 backbone
and expose additional arguments for fine-grained control of optimisation
hyper-parameters.

## References

Thanks to  [Davis](https://github.com/Tsingularity/FRN), [Phil](https://github.com/lucidrains/vit-pytorch) and  [Yassine](https://github.com/yassouali/SCL), for the preliminary implementations.

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:

- jijie@lut.edu.cn
