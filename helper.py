#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File containing all useful functions
"""

import matplotlib.image as mpimg
import random as rd
import numpy as np
import os,sys, csv, re
from PIL import Image
from scipy import ndimage
from math import ceil
import matplotlib.pyplot as plt
#----------------------- Functions to manipulate images --------------------------


def load_image(infilename):
    '''Load an image from filename'''
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    '''Converts values in [0,1] to "usual" RVB code in [|0, 255|]'''
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img, pixel_depth):
    '''Function to print images side-to-side for comparison (prediction, groundtruth, original image)'''
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        # Image in color
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        # Grey image
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8_(gt_img, pixel_depth)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8_(img, pixel_depth)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img, pixel_depth):
    '''Function to print both the original image and the predicted image, overlaid'''
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img*pixel_depth

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def img_crop(im, w, h):
    '''Turn an image into a list of patches'''
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


# ------------------------ CSV - related functions ----------------------------

def load_csv_data(data_path):
    '''Loads data and returns y (class labels), tX (features) and ids (event ids)'''
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1, usecols=0)
    y = y[:].astype(np.float)
    return y, x

def load_csv_dataimages(data_path, nbimages=50, nbpatchesx=38, nbpatchesy=38):
    '''Loads data and returns y (class labels), tX (features) and ids (event ids)'''
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1, usecols=0)
    y = y[:].astype(np.float)
    y=np.reshape(y,(nbimages,nbpatchesx,nbpatchesy))
    print(y.shape)
    return y

# ----------------------- Function manipulating submissions ----------------------------------

def brut_to_list(filename):
    '''Turns a csv file containing submission with float values in [0,1] into a list'''
    l = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first=True
        for row in csv_reader:
            if first :
                l.append(row)
                first = False
            else :
                l.append([row[0],float(row[1])])
    return l


def make_average(filename1, filename2, result_filename, coeff=0.5):
    '''Averages the two submission, with weight coeff for the second one and 1-weight for the first one'''
    l1 = brut_to_list(filename1)
    l2 = brut_to_list(filename2)
    with open(result_filename, 'w') as csvfile:
        fieldnames = ['id', 'prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, len(l1)) :
            label = (1-coeff)*l1[i][1]+coeff*l2[i][1]
            writer.writerow({'id': l1[i][0], 'prediction': label})

def filter_threshold(filename, threshold):
    '''Writes a submission filtering the brut data in filename using the given threshold. 
    New file has a similar name with "_threshold" in the end'''
    l = brut_to_list(filename)
    v_to_c = lambda x : int(x>threshold)
    with open(filename[:-4]+"_"+str(threshold)+".csv",'w') as filtered:
        wri = csv.writer(filtered, delimiter=',')
       
        wri.writerow(l[0])
        for row in l[1:]:
            wri.writerow([row[0],v_to_c(row[1])])
    filtered.close()    

        
        

# ----------------- Mask to submission functions -------------------------------------

def patch_to_label(patch, id_patch, foreground_threshold = 0.25):
    '''Assigns a label to a patch'''
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename, fct_predict=patch_to_label):
    '''Reads a single image and outputs the strings that should go into the submission file'''
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


def masks_to_submission(submission_filename, image_filenames):
    '''Converts images into a submission file'''
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


# ----------------- Submission to mask functions -------------------------------------

def img_float_to_uint8_(img, pixel_depth=3):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * pixel_depth).round().astype(np.uint8)
    return rimg


def reconstruct_from_labels(labels_filename):
    '''From a submission file (.csv), creates all corresponding grey-shaded images'''
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(labels_filename)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
    img_filenames= []
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j+w, imgwidth)
        ie = min(i+h, imgheight)
        if prediction == 0:
            adata = np.zeros((w,h))
        else:
            adata = np.ones((w,h))

        im[j:je, i:ie] = binary_to_uint8(adata)
    img_filename = label_file+'prediction_' + '%.3d' % image_id + '.png'
    img_filenames;append(img_filename)
    Image.fromarray(im).save(img_filename)

    return img_filenames

def binary_to_uint8(img):
    '''Convert an array of binary labels to a uint8'''
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def label_to_img(imgwidth, imgheight, w, h, labels):
    '''Convert array of labels to an image'''
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            l=labels[idx][1]
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels




def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])

# ---------------------- Functions for image processing   -----------------------------------

def rgb_to_grey(img):
    if len(img.shape)<3:
        return img
    imgGrey=0.3*img[:,:,0]+0.59*img[:,:,1]+0.11*img[:,:,2];
    return imgGrey


def compute_gradientNorm(img): #compute norm of gradient for each pixel
    imgGrey=rgb_to_grey(img)
    grad=np.zeros(imgGrey.shape)
    for i in range (imgGrey.shape[0]):
        for j in range (imgGrey.shape[1]):
            Gk=0
            Gl=0
            if i>0:
                Gl+=-np.sqrt(2)*imgGrey[i-1,j]
            if i>0 and j>0:
                Gl+=-imgGrey[i-1,j-1]
                Gk+=imgGrey[i-1,j-1]
            if i>0 and j<imgGrey.shape[1]-1:
                Gl+=-imgGrey[i-1,j+1]
                Gk+=-imgGrey[i-1,j+1]
            if i<imgGrey.shape[0]-1:
                Gl+=np.sqrt(2)*imgGrey[i+1,j]
            if i<imgGrey.shape[0]-1 and j>0:
                Gl+=imgGrey[i+1,j-1]
                Gk+=imgGrey[i+1,j-1]
            if i<imgGrey.shape[0]-1 and j<imgGrey.shape[1]-1:
                Gl+=imgGrey[i+1,j+1]
                Gk+=-imgGrey[i+1,j+1]
            if j>0:
                Gk+=np.sqrt(2)*imgGrey[i,j-1]
            if j<j<imgGrey.shape[1]-1:
                Gk+=-np.sqrt(2)*imgGrey[i,j+1]
            grad[i,j]=np.sqrt(Gl*Gl+Gk*Gk)
#    grad= (grad > 0.8)
    return grad #shape H*L*1


def compute_gradient(img): #compute gradient for each pixel
    imgGrey=rgb_to_grey(img)
    grad=np.zeros((imgGrey.shape[0],imgGrey.shape[1],2))
    for i in range (imgGrey.shape[0]):
        for j in range (imgGrey.shape[1]):
            Gk=0
            Gl=0
            if i>0:
                Gl+=-np.sqrt(2)*imgGrey[i-1,j]
            if i>0 and j>0:
                Gl+=-imgGrey[i-1,j-1]
                Gk+=imgGrey[i-1,j-1]
            if i>0 and j<imgGrey.shape[1]-1:
                Gl+=-imgGrey[i-1,j+1]
                Gk+=-imgGrey[i-1,j+1]
            if i<imgGrey.shape[0]-1:
                Gl+=np.sqrt(2)*imgGrey[i+1,j]
            if i<imgGrey.shape[0]-1 and j>0:
                Gl+=imgGrey[i+1,j-1]
                Gk+=imgGrey[i+1,j-1]
            if i<imgGrey.shape[0]-1 and j<imgGrey.shape[1]-1:
                Gl+=imgGrey[i+1,j+1]
                Gk+=-imgGrey[i+1,j+1]
            if j>0:
                Gk+=np.sqrt(2)*imgGrey[i,j-1]
            if j<j<imgGrey.shape[1]-1:
                Gk+=-np.sqrt(2)*imgGrey[i,j+1]
            grad[i,j,0]=Gl
            grad[i,j,1]=Gk
    return grad #shape H*L*2


def compute_second_derivative(img): #compute the second derivative for each pixel
    imgGrey=rgb_to_grey(img)
    grad=np.zeros((imgGrey.shape[0],imgGrey.shape[1],2))
    for i in range (imgGrey.shape[0]):
        for j in range (imgGrey.shape[1]):
            Gk=0
            Gl=0
            if i>0:
                Gl+=imgGrey[i-1,j]
            if i<imgGrey.shape[0]-1:
                Gl+=imgGrey[i+1,j]
            if j>0:
                Gk+=imgGrey[i,j-1]
            if j<j<imgGrey.shape[1]-1:
                Gk+=imgGrey[i,j+1]
            grad[i,j,0]=Gl-2*imgGrey[i,j]
            grad[i,j,1]=Gk-2*imgGrey[i,j]
    return grad #shape H*L*1


def compute_second_derivativeNorm(img): #compute the second derivative norm for each pixel
    imgGrey=rgb_to_grey(img)
    grad=np.zeros((imgGrey.shape[0],imgGrey.shape[1]))
    for i in range (imgGrey.shape[0]):
        for j in range (imgGrey.shape[1]):
            Gk=0
            Gl=0
            if i>0:
                Gl+=imgGrey[i-1,j]
            if i<imgGrey.shape[0]-1:
                Gl+=imgGrey[i+1,j]
            if j>0:
                Gk+=imgGrey[i,j-1]
            if j<j<imgGrey.shape[1]-1:
                Gk+=imgGrey[i,j+1]
            Gl=Gl-2*imgGrey[i,j]
            Gk=Gk-2*imgGrey[i,j]
            grad[i,j]=np.sqrt(Gl*Gl+Gk*Gk)
#    grad= (grad > 0.65)
    return grad #shape H*L*2

def compute_second_derivativergb(img): #compute the second derivative for each pixel
    imgR=img[:,:,0]
    imgG=img[:,:,1]
    imgB=img[:,:,2]
    gradR=compute_second_derivativeNorm(imgR)
    gradG=compute_second_derivativeNorm(imgG)
    gradB=compute_second_derivativeNorm(imgB)
    grad=np.zeros((img.shape[0],img.shape[1],3))
    grad[:,:,0]=gradR;
    grad[:,:,1]=gradG;
    grad[:,:,2]=gradB;
    return grad;


# ------------------ Extraction functions --------------------------------

def extract_data_all(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    w, h = imgs[0].shape[0], imgs[0].shape[1]
    nb_patches = (w/16)*(h/16)

    img_patches = [img_crop(imgs[i], 16, 16) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return np.asarray(data)

def value_to_class(v, threshold):
    # threshold percentage of pixels > 1 required to assign a foreground label to a patch, typically 0.25
    df = np.sum(v)
    if threshold==-1 :
        return [1-df, df] # No rounding
    elif df>threshold : 
        return [0,1]
    else : 
        return [1, 0]


def extract_labels(filename, image_indexes, threshold=-1):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in image_indexes:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], 16, 16) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i]), threshold) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)



# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img, grad_img, sec_img):
    # img is a patch (shape 16, 61, 3)
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    # feat [[average gray,variance gray]]    
    return feat

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features_RVB_6d(img, grad_img, sec_img):
    # img is a patch (shape 16, 61, 3)    
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

def extract_features_all_8d(img, grad_img, sec_img):
    '''RBG + grey variance and mean'''
    feat_gray = extract_features_2d(img, grad_img, sec_img)
    feat_color = extract_features_RVB_6d(img, grad_img, sec_img)
    feat = np.append(feat_gray,feat_color)
    return feat

def extract_features_subpatch(img, grad_img, sec_img):
    '''Features of each 4x4 subpatch'''
    feat = extract_features_RVB_6d(img, grad_img, sec_img)
    for i in range(4):
        for j in range(4):
            subpatch = img[i:i+2][j:j+2]
            feat  = np.append(feat,extract_features_RVB_6d(img, grad_img, sec_img))
    return feat

def extract_features_grad_norm(img, grad_img, sec_img):
    '''Norm of the average gradient vector'''
    feat_m = np.mean(grad_img)
    feat_v = np.var(grad_img)
    return np.append(feat_m, feat_v)

def extract_features_grad_norm_RVB(img, grad_img, sec_img):
    '''Norm of the average gradient vector'''
    feat = np.mean(grad_img)
    return np.append(feat, feat)

def extract_features_grad_only(img, grad_img, sec_img):
    feat_m = np.mean(grad_img)
    feat_v = np.var(grad_img)
    feat = np.append(feat_m, feat_v)
    return feat

def extract_features_grad_and_RVB(img, grad_img, sec_img):
    feat_RVB = extract_features_RVB_6d(img, grad_img, sec_img)
    feat_grad = extract_features_grad_only(img, grad_img, sec_img)
    feat = np.append(feat_RVB,feat_grad)
    return feat

# Extract features for a given image
def extract_img_features(filename, extractfct = extract_features_RVB_6d, features_needed=[]):
    img = load_image(filename)    
    img_patches = img_crop(img, 16, 16)
    if "grad" in features_needed :
        grads= compute_gradientT(img)
        grad_patches=  img_crop(grads, 16, 16)
    else : 
        grad_patches=  [0 for x in img_patches]
    if "sec" in features_needed :
        sec= compute_gradientT(img)
        sec_patches=  img_crop(sec, 16, 16)
    else : 
        sec_patches=  [0 for x in img_patches]
    X = np.asarray([extractfct(img_patches[i], grad_patches[i], sec_patches[i]) for i in range(len(img_patches))])
    return X



# ---------------------- Post -processing filter ------------------------------------------


def filter_postprocessing(csv_filename, output_filename, outputpath=""):
    '''Filters the data contained in the csv file and registers the filtered result in the output filename'''
    
    y=load_csv_dataimages(csv_filename)
    print(y.shape)

    plt.imshow(y[0], cmap="gray")


    MAT=np.array([[0,0,2,0,0],[0,1,2,1,0],[2,2,10,2,2],[0,1,2,1,0],[0,0,2,0,0]])
    MAT=MAT/np.sum(MAT)

    image_filenames = []
    for i in range (50):
        result=ndimage.convolve(y[i],MAT,mode='reflect')
        plt.imshow(result, cmap="gray") 
        result2=ndimage.zoom(result, 16, mode='nearest')           

        m=np.amax(result2)
        mM=np.amin(result2)
        result2=(result2-mM)/(m-mM)
        threshold=0.5
        resulti=result2>threshold
        plt.imshow(resulti, cmap="gray") 

        resulti = (resulti * 255).round().astype(np.uint8)
        
        image_filename = output_filename+'.png'
        image_filenames.append(image_filename)
        Image.fromarray(resulti).save(image_filename)
        
    masks_to_submission(output_filename, image_filenames)

    
