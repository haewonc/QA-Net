# QA-Net
### General
<p align="center"><img src="https://github.com/haewonc/QA-Net/blob/main/compare.jpeg" width="300px"></p>

In this repository, we provide a implementation of a quality map-associated attention network (QA-Net) for a multi-image super-resolution of a satellite images. The satellite images are often occluded by atmospheric disturbances such as clouds, and the position of the disturbances varies by the images. Many radiometric and geometric approaches are proposed to detect atmospheric disturbances. Still, the utilization of detection results, <i>i.e.</i>, quality maps in deep learning was limited to pre-processing or computation of loss. So we present an architecture that fully incorporates QMs into a deep learning scheme for the first time.

- [Preprint](https://arxiv.org/abs/2202.13124)
- [Supplementary Document](https://drive.google.com/file/d/1_ajvA3k8IUONUUs5oyMDCLLYiZh4lbZo/view?usp=sharing)
- [Competition Homepage](https://live.kelvins.esa.int/proba-v-super-resolution/)

<p align="center"><img src="https://github.com/haewonc/QA-Net/blob/main/arch.jpg" width="650px"></p>

### Setting
1. Clone this repository.
```
git clone https://github.com/haewonc/QA-Net.git
```
2. Match the dependencies.
```
conda env create -f requirements.txt
```
3. Download the pretrained model [here](https://drive.google.com/drive/folders/1MK2i-dPdFSm0vrU-sYXRNS8yikUdkErM?usp=sharing) and place it inside results/saved_models/.
   
### Preprocessing
1. Download cropped, splitted, and preprocessed dataset by [TR-MISR](https://github.com/Suanmd/TR-MISR). [Google Drive](https://drive.google.com/file/d/1_ZYJqHaXmAZqVlLVxLf118_R5wp7Rt7L/view)
2. Place the dataset outside the qa-net directory.
3. Process the validation and test set to select 9 low-resolution images. 
```
cd utils
python preprocess.py
```
Or you can just download our preprocessed dataset. [Google Drive](https://drive.google.com/drive/folders/1MK2i-dPdFSm0vrU-sYXRNS8yikUdkErM?usp=sharing)
### Test on validation set
1. Quantitative results
```
python valid.py
```
2. Qualitative results (TBU)

### Test on test set.
The ground-truth high-resolution images for test-set is not accessible. But you can generate submission file and submit it to the post-mortem leaderboard of the [competition](https://kelvins.esa.int/proba-v-super-resolution/). 
```
python predict.py
```

### Train QA-Net model
1. You can train QA-Net model with command below. The training with default configuration file should work on a machine with NVIDIA RTX 3080Ti (Memory: 12GB). 
```
python train.py
```
2. Check saved images during training in results/train_imgs and results/val_imgs.
3. You can change the data directory, number of modules, batch size, and total number of epochs by editing the `config.py`.
