import os
import nibabel as nib
import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice=0
    for index in range(numLabels):
        dice += dice_coefficient(y_true == index , y_pred == index)
    return dice/numLabels # taking average

def calculate_avg_dice(results_folder, ground_truth_folder):
    dice_scores = []

    for file_name in os.listdir(results_folder):
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            result_path = os.path.join(results_folder, file_name)
            gt_path = os.path.join(ground_truth_folder, file_name)

            if os.path.exists(gt_path):
                result_img = nib.load(result_path)
                gt_img = nib.load(gt_path)

                result_data = result_img.get_fdata()
                gt_data = gt_img.get_fdata()
                # print(gt_data.max(), result_data.max())

                dice = dice_coef_multilabel(gt_data, result_data, 4)
                dice_scores.append(dice)
            else:
                print(f"Ground truth for {file_name} not found!")

    if dice_scores:
        avg_dice = np.mean(dice_scores)
        print(f"Average Dice Coefficient: {avg_dice:.4f}")
    else:
        print("No valid segmentation files found for comparison.")

results_folder = 'outputs'
ground_truth_folder = 'Dataset137_BraTS2021_test/labelsTr'

calculate_avg_dice(results_folder, ground_truth_folder)