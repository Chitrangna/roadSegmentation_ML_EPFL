#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch, id_patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename, fct_predict=patch_to_label):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    id_patch = 0
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            if id_patch==0 : print(patch.shape)
            # i, j =  abscissa and ordinate of the upleft corner of the patch we're predicting
            label = fct_predict(patch,id_patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))
            id_patch+=1


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


if __name__ == '__main__':
    submission_filename = 'predictions_testing/mixedsubmission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = 'predictions_testing/mixedpredictionfiltered' +  str(i) + '.png'
        print (image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
