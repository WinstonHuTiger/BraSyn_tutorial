# BraSyn tutorial
This is the tutorial built for beginner to quickly get hands on the [BraSyn Challenge](https://www.synapse.org/#!Synapse:syn53708249/wiki/627507). The goal of this challenge is to generate one missing modality for a given MRI sequence. As for the goal of this tutorial is to help you get started and get comfortable with the challenge. Our primary tool is a simple 3D Generative Adversarial Network (GAN) known as pix2pix, but you are encouraged to experiment with more advanced models like Diffusion Models in your research.

## Dataset format 
Once you've downloaded and extracted the dataset from Synapse, take some time to understand its structure. Your folders for training and validation should look like this after extraction:

```
ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
├── BraTS-GLI-00000-000
├── BraTS-GLI-00002-000
├── BraTS-GLI-00003-000
├── BraTS-GLI-00005-000
... 
```

For each folder, there are 4 modalities available and one segmentation map (ending with ```*seg.nii.gz```):
```
BraTS-GLI-01666-000
├── BraTS-GLI-01666-000-seg.nii.gz
├── BraTS-GLI-01666-000-t1c.nii.gz
├── BraTS-GLI-01666-000-t1n.nii.gz
├── BraTS-GLI-01666-000-t2f.nii.gz
└── BraTS-GLI-01666-000-t2w.nii.gz
```

Tools like [itk-snap](http://www.itksnap.org/pmwiki/pmwiki.php?n=Main.HomePage) are useful to view each modality and segmentation map provided. After you view the input images, you can find the dimension of the images are all $256 \times 256 \times 256$, which contains too much empty space and is too big to train on memory limited GPUs. Therefore, we need to crop the images in latter process.

## The simple 3D baseline

The baseline model simulates a scenario where a random modality is missing during training, enhancing the model's ability to handle missing modalities during inference.

To get started, please clone this repo by:
```
git clone https://github.com/WinstonHuTiger/BraSyn_tutorial.git
```

### Environment setup

It is recommended to use [Mamba](https://mamba.readthedocs.io/en/latest/) for faster environment and package management compared to Anaconda. Install Mamba following the insctruction [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) and create a virtual environment as follows:
```
mamba create -n brasyn python=3.10
mamba activate brasyn
```

Alternatively, you can use Conda:

```
conda create -n brasyn python=3.10
conda activate brasyn
```


To install [Pytorch](https://pytorch.org/) on your machine with Cuda compatibility, you can just use the following command:

```
mamba install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```
or using conda:
```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

The minimum Pytorch version requirement is 1.9.0. If you want to install other version (your CUDA version is much lower than 11.8), please refer to [Pytorch installing docs](https://pytorch.org/get-started/previous-versions/). 


Then please install other dependencies by the following command:
```
cd project
pip install -r requirements.txt
```

### Inference
The inference pipeline can be summarized as following:
<image src="assets/inference_flow_chart.png" />

According the submission requirements, images are stored in a folder and model reads the processed images with cropped dimension ($144 \times 192 \times 192$) and generate the missing modality for the given input images. After the missing modality is generated, post-processed algorithm pads the images back to original dimension ($256 \times 256 \times 256 $).

To infer on your own machine, you have to do the following things:
- Run ```python drop_modality.py``` on your own machine to generate random missing modality MRI input sequence and please remember to change the ```val_set_folder``` to where you store your training dataset. 
- Change the ```data_path``` in ```project/generate_missing_modality.py``` to the same as the ```val_set_missing``` in ```drop_modality.py```.
- If you don't want to save the generated modality back to the data_path, change ```save_back``` in ```infer()``` function to ```False```
- Change the ```output_path``` in ```project/generate_missing_modality.py```, if you did the last step. 
- Run ```python project/generate_missing_modality.py``` to generate the missing modality.
**Note**: a **pre-trained** 3D GAN is given in ```mlcube/workspace/additional_files/weights/your_weight_name``` and parameter file is also included, ```mlcube/workspace/parameters.yaml```

After the inference, you need to obtain a Dice coefficient in order to evaluate your model further. 

**The most simple way** to do so is by using a nnUnet docker provided by us:
```
docker pull winstonhutiger/brasyn_nnunet:latest
```
Then you can just run the docker to obtain the Dice coefficient by using:
```
bash run_brasyn_nnunet_docker.sh
```
Alternatively, you can run the ensemble version to get more robust results:
```
docker pull winstonhutiger/brasyn_nnunet:ensemble
bash run_brasyn_nnunet_ensemble_docker.sh
```
**Note**: To run this docker, you have to [install Docker](https://docs.docker.com/engine/install/) on your own machine. Please pay attention to ```$PWD/pseudo_val_set``` in both ```bash``` scripts, which is the path where you store your generated modality and other **three true modalities**. Moreover, if you want to use post-processing in nnUnet ensemble docker, please add the ```--post``` flag after ```predict.sh```, changing the last line into  ```/bin/bash -c "bash predict.sh --post"```

<details>

<summary>Alternatively, you can also manually obtain the Dice score without docker.</summary> 
We provide a pretrained nnUnet for you to do so. There are several steps you should follow:

- Install nnUnetV2 on your machine, you can just use ```pip install nnunetv2``` to do so.
- Set the environment variable according to [the instruction here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md).
- Download the [pre-trained weight](https://drive.google.com/drive/folders/1dAKiXBpSQEthPZqELZ7snP2s9FIREBJk?usp=sharing) and put the unzipped folder to where you set ```nnUNet_results``` variable.
- Please use ```Dataset137_BraTS21.py``` to convert the generated missing modality and existing modality to nnunet's format (You have to change the path, ```brats_data_dir```, to where you store your MRI sequences).
- Run nnunetv2 by ```nnUNetv2_predict -i "./Dataset137_BraTS2021_test/imagesTr" -o "./outputs"  -d 137 -c 3d_fullres -f 5``` to obtain the predicted segmentation maps.
- Finally, you can use ```python cal_avg_dice.py``` to calculate the average Dice score, in order to evaluate your model on training dataset.
</details>

## Building MLCube

The detailed document for building model MLCube can be found [here](https://docs.medperf.org/mlcubes/mlcube_models/).

Please follow the [document here](https://docs.medperf.org/getting_started/installation/) to install Medperf.

The files needed to build model MLCube are all included in this repo already.


 **All you need to do**:
 - Change the image name and author name in ```mlcube/mlcube.yaml```
 - Change the name in ```mlcube/workspace/parameters.yaml``` to match the **weight folder' name**
 - Move your trained weight folder to the ```mlcube/workspace/additional_files``` folder
 - Move the ```checkpoint``` folder out of the ```project``` folder. 

After these steps, you can use:
```
mlcube configure -Pdocker.build_strategy=always
```
to build your MLCube on your machine. 

**Note: the logic for testing MLCube using [MedPref](https://www.medperf.org) and testing your model using nnunet provided in this repo is different.** In this repo, we copy the other three modalities and the generated modality to a subfolder (with patient name), within the main folder ```pseudo_val_set``` for the nnunet to segment. However, while using [MedPref](https://www.medperf.org) for MLCube submission, you only save the generated modality to a folder **without subfolders**. 

## Submitting MLCube
Please follow the detailed [description available here](https://www.synapse.org/#!Synapse:syn53708249/wiki/627758) to submit your MLCube.

 **Note**: **Do not forget** to test the compatibility using [MedPref](https://www.medperf.org) before submission.

 ## Training the baseline
The whole framework is built on [2D_VAE_UDA_for_3D_sythesis](https://github.com/WinstonHuTiger/2D_VAE_UDA_for_3D_sythesis) with a few tweaks from last year's BraSyn challenge. Compared with the original implementation, a new dataloader named ```brain_3D_random_mod_dataset.py``` is added to ```data``` folder. Input 3D volumes are manually cropped into sub-volumes with size $144 \times 192 \times 192$. For inference purpose, ```generate_missing_modality_options.py``` is added to ```option``` folder and some utility functions in ```generate_missing_modality.py``` are included to pad the output volumes for MLCube production. 


If you are interested in training your own simple baseline model, you first have to start the visdom server on your machine by ```visdom``` and then you can modify the following command:

```
cd project/
python train.py \
     --dataroot your_dataset_root_path \
     --name your_checkpoint_name \
     --model pix2pix --direction AtoB --dataset_mode brain_3D_random_mod \
     --B_modality random \
     --n_epochs 50 --n_epochs_decay 70 \
     --lr 0.0001 --input_nc 3 --output_nc 1 \
     --paired --netG sit --netD n_layers --n_layers_D 1 \
     --batch_size 1 --gpu_ids 0 
```
**Note**: the minimum GPU memory requirement is 24GB and the training time is about 28 hours on a single 4090 RTX GPU.
After the training, you have able to view Structural Similarity (SSIM) Index and Peak Signal-to-Noise Ratio (PSNR)  metric in the output. SSIM indicates the structural similarity, such as tissue similarity in our case here. As for the segmentation (Dice) score, we will discuss it in [inference](#inference). You are also welcome to include other metrics in your own research. 

 ## Citation
 Please cite our work, if you find this tutorial is somehow useful.
```
@misc{li2023brain,
      title={The Brain Tumor Segmentation (BraTS) Challenge 2023: Brain MR Image Synthesis for Tumor Segmentation (BraSyn)}, 
      author={Hongwei Bran Li and Gian Marco Conte and Syed Muhammad Anwar and Florian Kofler and Ivan Ezhov and Koen van Leemput and Marie Piraud and Maria Diaz and Byrone Cole and Evan Calabrese and Jeff Rudie and Felix Meissen and Maruf Adewole and Anastasia Janas and Anahita Fathi Kazerooni and Dominic LaBella and Ahmed W. Moawad and Keyvan Farahani and James Eddy and Timothy Bergquist and Verena Chung and Russell Takeshi Shinohara and Farouk Dako and Walter Wiggins and Zachary Reitman and Chunhao Wang and Xinyang Liu and Zhifan Jiang and Ariana Familiar and Elaine Johanson and Zeke Meier and Christos Davatzikos and John Freymann and Justin Kirby and Michel Bilello and Hassan M. Fathallah-Shaykh and Roland Wiest and Jan Kirschke and Rivka R. Colen and Aikaterini Kotrotsou and Pamela Lamontagne and Daniel Marcus and Mikhail Milchenko and Arash Nazeri and Marc André Weber and Abhishek Mahajan and Suyash Mohan and John Mongan and Christopher Hess and Soonmee Cha and Javier Villanueva and Meyer Errol Colak and Priscila Crivellaro and Andras Jakab and Jake Albrecht and Udunna Anazodo and Mariam Aboian and Thomas Yu and Verena Chung and Timothy Bergquist and James Eddy and Jake Albrecht and Ujjwal Baid and Spyridon Bakas and Marius George Linguraru and Bjoern Menze and Juan Eugenio Iglesias and Benedikt Wiestler},
      year={2023},
      eprint={2305.09011},
      archivePrefix={arXiv},
```
```
@misc{https://doi.org/10.48550/arxiv.2207.00844,
  author = {Hu, Qingqiao and Li, Hongwei and Zhang, Jianguo},  
  title = {Domain-Adaptive 3D Medical Image Synthesis: An Efficient Unsupervised Approach},
  publisher = {arXiv},
  year = {2022},
}

```

 ## Aknowledgement
The 3D synthesis baseline is from [2D_VAE_UDA_for_3D_sythesis](https://github.com/WinstonHuTiger/2D_VAE_UDA_for_3D_sythesis). 

The 3D backbone is from [3D-MRI-style-transfer](https://github.com/KreitnerL/3D-MRI-style-transfer).

The training framework is from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

Some utility functions are from Bran's [BraSyn](https://github.com/hongweilibran/BraSyn).

Thanks for their wonderful opensource works!!!
