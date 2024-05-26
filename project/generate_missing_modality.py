
from util.crop_and_pad_volume import crop_or_pad_volume_to_size_along_x, crop_or_pad_volume_to_size_along_y, crop_or_pad_volume_to_size_along_z
import os
# from options.generate_missing_modality_options import GenMissingOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_3D_images, save_images
from util import html
from util.ssim import ssim
from tqdm import tqdm
import numpy as np
from scipy.stats import wilcoxon
from models.networks import setDimensions
from torchvision import transforms
from data.data_augmentation_3D import  PadIfNecessary, getBetterOrientation, toGrayScale
# from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import nibabel as nib
from data.image_folder import get_available_3d_vol_names, POST_FIXES
for_segmentation_names = {"t1c": "t1ce", "t1n": "t1", "t2f": "flair", "t2w": "t2"}

class OPTIONS:
    def __init__(self) -> None:
    
        self.dataroot = ""
        self.output_dir = ""

def convert_image_to_256_3d(x):
        # x, [1, 1, 144, 192, 192]
        
    #batch_size = 1
    #channels = 1
    shape = x.shape
    x = x.squeeze()
    x = crop_or_pad_volume_to_size_along_x(x, 240)
    x = crop_or_pad_volume_to_size_along_y(x, 240)
    x = crop_or_pad_volume_to_size_along_z(x, 155)
    return torch.unsqueeze(torch.unsqueeze(x, 0), 0) # x, [1,1, 256, 256, 256]


def save_to_output(x, affine,  A1_path, output_target_path, test_target_modality, for_segmentation = False):
    # pass 
    x = x.squeeze()
    img = nib.Nifti1Image(x.numpy(), affine.squeeze().numpy())
    patient_name = A1_path.split('/')[-2]
    # os.makedirs(os.path.join( output_target_path), exist_ok=True)
    # print("img shape", img.shape)
    if for_segmentation:
        test_target_modality = "brain" + "_" + for_segmentation_names[test_target_modality]
    
        nib.save(img, os.path.join( output_target_path, patient_name,
                patient_name + "_" + test_target_modality + ".nii.gz") )
    else:
        out_path = os.path.join( output_target_path,
                 patient_name + "-" + test_target_modality + ".nii.gz")
        # print(out_path)
        nib.save(img,  out_path )

def infer(data_path, output_path, parameters_file, weights, save_back = False):
     # get test options
    # hard-code some parameters for test
    # opt = GenMissingOptions().parse()  # get test options
    opt = OPTIONS()
    opt.dataroot = data_path 
    opt.output_dir = output_path
    for key in parameters_file.keys():
        setattr(opt, key, parameters_file[key])
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.paired = True
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.checkpoints_dir = weights
    os.makedirs(opt.output_dir, exist_ok=True)
    
    # create a webpage for viewing the results
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if not opt.single_folder:
        opt.phase = 'test'
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)      # create a model given opt.model and other options
        print("dataset size: ", len(dataset), 'dataset root: ', dataset.dataset.root)
        # print(dataset.dataset.root)
        for i, data in enumerate(tqdm(dataset, total=min(opt.num_test, len(dataset)), desc='Testing')):
            if i == 0:
                
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
                if opt.eval:
                    model.eval()
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            # input = np.squeeze( data["B"].cpu().numpy())
            # output = np.squeeze(model.fake_B.cpu().numpy())
            output = model.fake_B.cpu()
            
            output_cube = convert_image_to_256_3d(output).squeeze()
            # print(data['A_paths'])
            # print(data['test_target_modality'])
            affine_matrix = data['affine']
            # print(output_cube.shape)
            if save_back:
                folder_path = None 
                if 'folder_path' in data.keys():
                    folder_path = data['folder_path'][0]
                else:
                    a_path = data['A_paths'][0]
                    temp = a_path.split(os.path.sep)
                    folder_path = os.path.join(*temp[:-1])
                # print(folder_path[0])
                    # folder_path = os.path.join(data['A_paths'][0], "..")
                save_to_output(output_cube,
                           affine_matrix , 
                           data['A_paths'][0], 
                           folder_path, data['test_target_modality'][0],
                           for_segmentation = opt.for_segmentation)
            else:
                save_to_output(output_cube,
                           affine_matrix , 
                           data['A_paths'][0], 
                           opt.output_dir, data['test_target_modality'][0],
                           for_segmentation = opt.for_segmentation)
            
    else:
        pass
    
if __name__ == "__main__":
    import yaml
    data_path = "pseudo_val_set"
    output_path = "pseudo_val_output"
    weights = "mlcube/workspace/additional_files/weights/"
    parameters_file = "mlcube/workspace/parameters.yaml"
    with open(parameters_file) as f:
        parameters = yaml.safe_load(f)
    infer(data_path, output_path, parameters, weights, save_back= True)