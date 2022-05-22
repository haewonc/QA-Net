# QA-Net
### Setting
1. Clone this repository.
```
git clone https://github.com/haewonc/qanet.git
```
2. Match the dependencies.
3. Download the pretrained model [here]() and place it inside results/saved_models/.
   
### Preprocessing
1. Download cropped, splitted, and preprocessed dataset by [TR-MISR](https://github.com/Suanmd/TR-MISR). [Google Drive](https://drive.google.com/file/d/1_ZYJqHaXmAZqVlLVxLf118_R5wp7Rt7L/view)
2. Place the dataset outside the qa-net directory.
3. Process the validation and test set to select 9 low-resolution images. 
```
cd utils
python preprocess.py
```

### Test on validation set
1. Qualitative results
```
python valid.py
```
2. Quantitative results (TBU)

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