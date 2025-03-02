# SYLBER: Syllabic Embedding Representation of Speech from Raw Audio

[Paper](https://arxiv.org/abs/2410.07168) | [Audio Samples](https://berkeley-speech-group.github.io/sylber)


## Updates

### 03/02/2025
1. Distribute inference package

### 12/25/2024
1. Initial code release with training and inference pipelines.
2. Checkpoint release

## Installation

The model can be installed through pypi for inference. 

```
pip install sylber
```
Please check [demo notebook](demo.ipynb) for the usage.
For training, please follow the below instructions.

## Environment

Install the dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Training SYLBER

### Datasets and Checkpoints

1. **Noise Dataset for WavLM-based Augmentation**: The noise dataset for the WavLM noise augmentation is sourced from [DNS Challenge](https://github.com/microsoft/DNS-Challenge). You can use the following script to download the dataset:
   ```
   bash download-dns-challenge-3.sh
   ```
    and untar `datasets_fullband/datasets_fullband.noise_fullband.tar.bz2`

2. **Generated Datasets**: The other data used for training SYLBER are generated using the [SDHuBERT repository](https://github.com/cheoljun95/sdhubert). Please follow the instructions there for data preparation.

3. **Checkpoints**: Pretrained model checkpoints for sylber are available on Google Drive: [link](https://drive.google.com/drive/folders/1Savigp6jnLKIAZ-6nwKkIwJ8Us1CkG5e?usp=sharing)

### Stage 1 Training
```bash
python train.py --config-name=sylber_base
```

### Stage 2 Training
```bash
python train.py --config-name=sylber_base_stage2
```

The training is split into two stages. Make sure to review the configurations in the `configs/` directory for detailed settings.

## Inference

### Segmentation and Visualization

For inference to obtain segmentations and visualize results, please refer to `demo.ipynb`.

### SPARC (formerly known as Articulatory Encodec)

For using SPARC, refer to [Speech-Articulatory-Coding](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding) for installation and usage instructions.



## Acknowledgements

Website adapted from: https://github.com/BytedanceSpeech/bytedancespeech.github.io

## Citation

If you use this work, please cite our paper:
```
@article{cho2024sylber,
  title={Sylber: Syllabic Embedding Representation of Speech from Raw Audio},
  author={Cho, Cheol Jun and Lee, Nicholas and Gupta, Akshat and Agarwal, Dhruv and Chen, Ethan and Black, Alan W and Anumanchipalli, Gopala K},
  journal={arXiv preprint arXiv:2410.07168},
  year={2024}
}
```
