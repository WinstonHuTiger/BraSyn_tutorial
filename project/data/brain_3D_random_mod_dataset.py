from data.base_dataset import BaseDataset
from data.image_folder import get_available_3d_vol_names, POST_FIXES
import nibabel as nib
import random
from torchvision import transforms
import os
import numpy as np
import torch
from models.networks import setDimensions
import skimage.transform as sk_trans
from data.data_augmentation_3D import ColorJitter3D, PadIfNecessary, SpatialRotation, SpatialFlip, getBetterOrientation, toGrayScale

class brain3DRandomModDataset(BaseDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--dataset_name', type=str, default="center_1_2", help='new dataset option')
        
        # parser.add_argument("--train_number", type=int, default=10000000, help="the training number of the dataset")
        # parser.add_argument("--k_fold", type = int, default = 0, help = "the fold parameter, K")
        
        
        # parser.add_argument("--datast", action="store_true", help="use paired modality training method")
        parser.add_argument("--B_modality", type = str, choices = ["random"] + POST_FIXES,  help="the target modality for training")
        parser.add_argument("--train_number", type=int, default=10000000, help="the training number of the dataset")
        parser.add_argument("--get_mask", action = "store_true", help = "use mask for training")
        
        # parser.add_argument("--dataset_name", type=str, default="center_1_2", help="the name of the dataset")
        
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser
    
    def __init__(self, opt):
        super().__init__(opt)
        # self.A1_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 't1', opt.phase), 't1.nii.gz'))
        # self.A2_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'flair', opt.phase), 'flair.nii.gz'))
        # self.B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 'dir', opt.phase), 'dir.nii.gz'))
        # suffix_str = ""
        # if opt.k_fold > 0:
            
        #     suffix_str = "_" + str(opt.k_fold) + "_fold"
        # self.A1_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 
        #                                                                 opt.dataset_name+ "_" + str(opt.phase) + suffix_str), 't2.nii.gz'))
        # self.A2_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 
        #                                                                 opt.dataset_name+ "_" + str(opt.phase)+ suffix_str), 'flair.nii.gz'))
        # self.B_paths = natural_sort(get_custom_file_paths(os.path.join(opt.dataroot, 
        #                                                                opt.dataset_name + "_" + str(opt.phase)+ suffix_str), 't1.nii.gz'))
        
        # self.A_size = len(self.A1_paths)  # get the size of dataset A
        # self.B_size = len(self.B_paths)  # get the size of dataset B
        
        if self.opt.phase == 'train':
            self.dataset_name = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        elif self.opt.phase == 'val':
            self.dataset_name = "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
        elif self.opt.phase == 'test':
            self.dataset_name = ""
        
        self.data_root = os.path.join(opt.dataroot, self.dataset_name)
        self.image_folders  = os.listdir(self.data_root)
        for folder in self.image_folders:
            if not os.path.isdir(os.path.join(self.data_root, folder)):
                self.image_folders.remove(folder)
        
        self.with_mask = opt.get_mask
        
        setDimensions(3)
        # 3 input dimension
        opt.input_nc = 3
        opt.output_nc = 1

        transformations = [
            transforms.Lambda(lambda x: getBetterOrientation(x, "IPL")),
            transforms.Lambda(lambda x: np.array(x.get_fdata())[np.newaxis, ...]),
            # transforms.Lambda(lambda x: sk_trans.resize(x, (256, 256, 160), order = 1, preserve_range=True)),
            # image size [1, 160, 240, 240]
            transforms.Lambda(lambda x: x[:,8:152,24:216,24:216]),
            # transforms.Lambda(lambda x: resize(x, (x.shape[0],), order=1, anti_aliasing=True)),
            transforms.Lambda(lambda x: toGrayScale(x)),
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float16 if opt.amp else torch.float32)),
            # transforms.Resize((256, 256)),
            PadIfNecessary(3),
        ]
        

        if(opt.phase == 'train'):
            self.updateTransformations += [
                SpatialRotation([(1,2), (1,3), (2,3)], [*[0]*12,1,2,3], auto_update=False), # With a probability of approx. 51% no rotation is performed
                SpatialFlip(dims=(1,2,3), auto_update=False)
            ]
        transformations += self.updateTransformations
        self.transform = transforms.Compose(transformations)
        self.colorJitter = ColorJitter3D((0.3,1.5), (0.3,1.5))
        
    

    def __getitem__(self, index):
        vol_folder = self.image_folders[index % len(self.image_folders)]
        
        vol_names, mask_path = get_available_3d_vol_names(os.path.join(self.data_root, vol_folder),self.with_mask)
        
        vols = {}
        
        affines = {}
        test_target_modality = ""
        for name in vol_names:
            if vol_names[name] is None:
                test_target_modality = name
                continue
            vols[name] = nib.load(os.path.join( self.data_root ,vol_folder, vol_names[name]))
            affines[name] = vols[name].affine
            
            
        B = None 
        target_modality = ""
        if self.opt.phase == 'train' or self.opt.phase == 'val':
            if self.opt.B_modality != "random":
                target_modality = self.opt.B_modality
                # B = vols[self.opt.B_modality]
            else:
                target_modality = random.choice(POST_FIXES)
                # B = vols[target_modality]
        
        A_keys = list(vols.keys())
        B_path = "" 
        
        # for training and validation, we need to select a target modality
        if target_modality in A_keys:
            B = self.transform(vols[target_modality])
            B_path = os.path.join( self.data_root ,vol_folder, vol_names[target_modality])
            A_keys.remove(target_modality)
            
        # apply transformation to the rest of the modalities
        As = []
        count = 0
        for key in A_keys:
            temp =  self.transform(vols[key])
            if self.opt.phase == 'train':
                temp = self.colorJitter(temp, no_update=True if count > 0 else False)
                count += 1
            As.append(temp)

        # fix the bug with affine matrix
        affine = affines[A_keys[0]]
        targ_ornt = nib.orientations.axcodes2ornt("IPL")
        affine = nib.orientations.ornt_transform(affine, targ_ornt)
        
        A = torch.concat(As, dim=0)
        # print("A shape: ", A.shape)
        # print("B shape: ", B.shape)
        # print("target modality: ", target_modality)
        A1_path = os.path.join( self.data_root, vol_folder, vol_names[A_keys[0]])
        
        mask = -1
        if self.with_mask and mask_path is not None:
            mask = self.transform(nib.load(os.path.join( self.data_root,vol_folder, mask_path)))            
        
        if B is not None:
            return {'A': A, 
                    'B': B,
                    'affine': affine, 
                    'axis_code': "IPL",
                    'A_paths': A1_path, 'B_paths': B_path, 
                    'test_target_modality': test_target_modality,
                    'mask': mask}
        else:
            return {'A': A,
                    "B": A,
                    'affine': affine,
                    'axis_code': "IPL",
                    'A_paths': A1_path, 
                    "folder_path": os.path.join( self.data_root, vol_folder),
                    'test_target_modality': test_target_modality
                    }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return min(self.opt.train_number, len(self.image_folders))
