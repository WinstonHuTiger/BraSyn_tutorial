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

Tools like [itk-snap](http://www.itksnap.org/pmwiki/pmwiki.php?n=Main.HomePage) are useful to view each modality and segmentation map provided. After you view the input images, you can find the dimension of the images are all $256 \times 256 \times 256$, which contains too much empty space and is too big to train on memory limited GPUs. Therefore, we need to crop the images in latter process.

## The weak 3D baseline

The baseline is built on [2D_VAE_UDA_for_3D_sythesis](https://github.com/WinstonHuTiger/2D_VAE_UDA_for_3D_sythesis) with a few tweaks from last year's BraSyn challenge The baseline model simulates a scenario where a random modality is missing during training, enhancing the model's ability to handle missing modalities during inference.

Compared with the original implementation, a new dataloader named ```brain_3D_random_mod_dataset.py``` is added to ```data``` folder. Input 3D volumes are manually cropped into sub-volumes with size $144 \times 192 \times 192$. For inference purpose, ```generate_missing_modality_options.py``` is added to ```option``` folder and some utility functions in ```generate_missing_modality.py``` are included to pad the output volumes for MLCube production. 


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

### Training

To start training the baseline, you first have to start the visdom server on your machine by ```visdom``` and then you can modify the following command:

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
After the training, you have able to view Structural Similarity (SSIM) Index and Peak Signal-to-Noise Ratio (PSNR)  metric in the output. SSIM indicates the structural similarity, such as tissue similarity in our case here. As for the segmentation score, we will discuss it in [inference](#inference). You are also welcome to include other metrics in your own research. 

### Inference
The inference pipeline can be summarized as following:
<image src="assets/inference_flow_chart.png" />

According the submission requirements, images are stored in a folder and model reads the processed images with cropped dimension and generate the missing modality for the given input images. After the missing modality is generated, post-processed algorithm pads the images back to original dimension.

To do the infer on your own machine, you have to do the following:
- Run ```drop_modality.py``` on your own machine to generate random missing modality MRI inpout sequence. 
- Change the data_path in ```project/generate_missing_modality.py```
- If you want to save the generated modality back to the data_path, change ```save_back``` in ```infer()``` function to ```True```
- Change the output_path in ```project/generate_missing_modality.py```
**Note**: a **pre-trained** 3D GAN is given in path ```mlcube/workspace/additional_files/weights/your_weight_name``` and parameter file is also given in ```mlcube/workspace/parameters.yaml```

After the inference, you can use a pre-trained nnUnet to obtain the Dice score (if you save your generated missing modality back to data_path), we will add the instruction latter here. 

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

Some utility functions are from Bran's [BraSyn](https://github.com/hongweilibran/BraSyn).

Thanks for their wonderful opensource works!!!
