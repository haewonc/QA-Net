# QA-Net
### General
<p align="center"><img src="https://cvws.icloud-content.com/B/AXWQF-si_RdiZ0w3EVdlIjScrIwPAevW1O4IKfrDo-w9d2hIuL2knuN2/compare.jpeg?o=AkGfh0dY85k0o9AUvZUTzozhNdJwz1gYIwIFgAM4k-kS&v=1&x=3&a=CAogkm_-mEYWFwjmziy9MMp9nBBQqNbJaoBviLwsvQrAxskSbxCD9NWpqjAYg9Gxq6owIgEAUgScrIwPWgSknuN2aifkIZjB-5kAdMIn5vTXWprWgnSjEQG9ybHA4sGGyqWiGbZxayTTixtyJz1pd12TrLuF_HnzuGdLXxx9qXk0gE63Wmgh7SeUiBjTNOmjyZb2KA&e=1660632721&fl=&r=7e8fa8ec-dcf7-4dc1-b563-b900f7c6c0e3-1&k=dgDen2GXT8OXzTBH3ZajpQ&ckc=com.apple.clouddocs&ckz=com.apple.CloudDocs&p=112&s=7dFSjsXmgJLLeIYb4TKB1HR3Gi8&cd=i" width="300px"></p>

In this repository, we provide a implementation of a quality map-associated attention network (QA-Net) for a multi-image super-resolution of a satellite images. The satellite images are often occluded by atmospheric disturbances such as clouds, and the position of the disturbances varies by the images. Many radiometric and geometric approaches are proposed to detect atmospheric disturbances. Still, the utilization of detection results, <i>i.e.</i>, quality maps in deep learning was limited to pre-processing or computation of loss. So we present an architecture that fully incorporates QMs into a deep learning scheme for the first time.

- [Preprint](https://arxiv.org/abs/2202.13124)
- [Supplementary Document](https://drive.google.com/file/d/1_ajvA3k8IUONUUs5oyMDCLLYiZh4lbZo/view?usp=sharing)
- [Competition Homepage](https://live.kelvins.esa.int/proba-v-super-resolution/)

<p align="center"><img src="https://cvws.icloud-content.com/B/AUL5eoQ8gsv6200yPS_sk6ipUuQqASaIM2EFi_18AA7yYpIjMiUTRuYt/arch.jpg?o=AhGbG7r9xsD0mbCcpfkSv09WLzLoZXOyIZ8bj48nWRFs&v=1&x=3&a=CAogW4HT9mg2mn0Iiuq3KVSiIeuPbdsXe5xsfJyUFWjBsDoSbxCby_apqjAYm6jSq6owIgEAUgSpUuQqWgQTRuYtaievdNA6GZSm1xb_AaicyuY9-kQksVLtiSvzfaBePwmQrgIZ2MkZeQhyJ3cQlUjAdtlXQRMQGZXD4Lu1QoFjf4bn66HD_vv0H0Qcs8P7dCTtFg&e=1660633256&fl=&r=1a3cd5af-3775-4c97-b67f-0aba0c1a865a-1&k=pltRNYEGEpVATxVPdcKsWg&ckc=com.apple.clouddocs&ckz=com.apple.CloudDocs&p=112&s=-RjdAWx5yCYBqY57bWbLnQyO5BQ&cd=i" width="650px"></p>

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
