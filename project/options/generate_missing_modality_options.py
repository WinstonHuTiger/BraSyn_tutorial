from .base_options import BaseOptions


class GenMissingOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--output_dir', type=str, default='./output_dir', help='saves outputs here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--single_folder', action='store_true', help='use single folder for testing')
        # parser.add_argument('--whole_dataset', action ="store_true", help = "use whole dataset for testing")
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--no_nifti', action='store_true', help='do not generate nifti files in 3D mode')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser