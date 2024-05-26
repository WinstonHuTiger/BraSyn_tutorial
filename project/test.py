"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_3D_images, save_images
from util import html
from util.ssim import ssim
from tqdm import tqdm
import numpy as np
from scipy.stats import wilcoxon
# from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
def get_psnr(x, y, data_range):
    EPS = 1e-8

    x = x / float(data_range)
    y = y / float(data_range)

    # if (x.size(1) == 3) and convert_to_greyscale:
    #     # Convert RGB image to YCbCr and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
    #     rgb_to_grey = torch.tensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1).to(x)
    #     x = torch.sum(x * rgb_to_grey, dim=1, keepdim=True)
    #     y = torch.sum(y * rgb_to_grey, dim=1, keepdim=True)

    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3, 4])
    score: torch.Tensor = - 10 * torch.log10(mse + EPS)
    return torch.mean(score, dim = 0)



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.paired = True
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    ssim_list = []
    psnr_list = []
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

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
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        # input = np.squeeze( data["B"].cpu().numpy())
        # output = np.squeeze(model.fake_B.cpu().numpy())
        input = data["B"].cpu()
        output = model.fake_B.cpu()
        
        ssim_list.append(ssim(input, output))
        psnr_list.append(get_psnr(input, output , data_range=output.max() - output.min()))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
        if data['A'].dim() == 5 and not opt.no_nifti:
            save_3D_images(webpage, model.get_current_visuals(slice=False), img_path, data['affine'][0].cpu().numpy(), data['axis_code'][0])
    webpage.save()  # save the HTML
    
    print(opt.phase, " average SSIM: ", np.mean(ssim_list))
    print(opt.phase, " average PSNR: ", np.mean(psnr_list))
    
    print(opt.phase, " std of SSIM: ", np.std(ssim_list))
    print(opt.phase, " std of PSNR: ", np.std(psnr_list))
    
    print(opt.phase, " p-value of SSIM: ", wilcoxon(ssim_list)[-1])
    print(opt.phase, " p-value of PSNR: ", wilcoxon(psnr_list)[-1])