# BraSyn tutorial
This is the tutorial built for beginner to quickly get hands on the BraSyn Challenge. The goal of this challenge is to generate one missing modality for a given MRI sequence. As for the goal of this tutorial is to help you get started and get comfortable with the challenge. Our primary tool is a simple 3D Generative Adversarial Network (GAN) known as pix2pix, but you are encouraged to experiment with more advanced models like Diffusion Models in your research.

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

For each folder, there are 4 modalities available and one segmentation map:
```
BraTS-GLI-01666-000
├── BraTS-GLI-01666-000-seg.nii.gz
├── BraTS-GLI-01666-000-t1c.nii.gz
├── BraTS-GLI-01666-000-t1n.nii.gz
├── BraTS-GLI-01666-000-t2f.nii.gz
└── BraTS-GLI-01666-000-t2w.nii.gz
```

Tools like [itk-snap](http://www.itksnap.org/pmwiki/pmwiki.php?n=Main.HomePage) are useful to view each modality and segmentation map provided. 

## The weak 3D baseline
In this repo, we provide a weak 3D baseline from [Qingqiao Hu](https://winstonhutiger.github.io/) from last year's BraSyn challenge. 


To get started, please clone this repo by:
```
git clone 
```

### Environment setup

It is recommended to use [Mamba](https://mamba.readthedocs.io/en/latest/) for faster environment and package management compared to Anaconda. Install Mamba following the insctruction [here]((https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)) and create a virtual environment as follows:
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

The minimum Pytorch version requirement is 1.9.0. If you want to install other version, please refer to [Pytorch installing docs](https://pytorch.org/get-started/previous-versions/).


Then please install other dependencies by the following command:
```
cd project
pip install -r requirements.txt
```

### Training
The baseline is built on [2D_VAE_UDA_for_3D_sythesis](https://github.com/WinstonHuTiger/2D_VAE_UDA_for_3D_sythesis) with a few tweaks. The baseline model simulates a scenario where a random modality is missing during training, enhancing the model's ability to handle missing modalities during inference.

Compared with the original implementation, a new dataloader named ```brain_3D_random_mod_dataset.py``` is added to ```data``` folder. Input 3D volumes are manually cropped into sub-volumes with size $144 \times 192 \times 192$. For inference purpose, ```generate_missing_modality_options.py``` is added to ```option``` folder and some utility functions in ```generate_missing_modality.py``` are included to pad the output volumes for MLCube production. 


To get started the baseline training, you first have to start the visdom server on your machine by ```visdom``` and then you can modify the following command:

```
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

## Building MLCube

The detailed document for building model MLCube can be found [here](https://docs.medperf.org/mlcubes/mlcube_models/).

Please follow the [document here](https://docs.medperf.org/getting_started/installation/) to install Medperf.

The files needed to build model MLCube are all included in this repo already.

 **All you need to do**:
 - Change the image name and author name in ```mlcube/mlcube.yaml```
 - Change the name in ```mlcube/workspace/parameters.yaml``` to match the **weight folder' name**
 - Move your trained weight folder to the ```mlcube/workspace/additional_files``` folder
 - Move the ```checkpoint``` folder out of the ```project``` folder. 

After those two steps, you can use:
```
mlcube configure -Pdocker.build_strategy=always
```
to build your MLCube on your machine. 

## Submitting MLCube
Please follow the detailed [description available here](https://www.synapse.org/#!Synapse:syn53708249/wiki/627758) to submit your MLCube.

 **Note**: **Do not forget** to test the compatibility before submission.

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
Some utility function are from Bran's [BraSyn](https://github.com/hongweilibran/BraSyn).
Thanks for their wonderful opensource works!!!
