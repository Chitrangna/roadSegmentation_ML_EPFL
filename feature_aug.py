"""
File realizing feature augmentation. Running it will create more images in the training set and the corresponding groundtruth.
It uses rotations and symmetries (central, horizontal axis or vertical axis)

"""

import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import random as rd

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def rotation(img, isnt_groundtruth) :
    '''Rotation of -pi/4'''
    x = np.zeros(img.shape)
    columns, lines = img.shape[0], img.shape[1]
    
    for i in range(0,lines):
        for j in range(0,columns):
            # 3 dimensions
            if isnt_groundtruth :
                for c in range(0, 3):
                    x[i,j,c]=img[columns-j-1,i,c]
            # 1 dimension
            else : 
                x[i,j] = img[columns-j-1,i]     
    return x


def mirror(img, axis, isnt_groundtruth) :
    '''Mirror with axis horizontal if axis = 1, vertical if axis = 2 and central if axis = 0 '''
    x = np.zeros(img.shape)
    lines, columns = img.shape[0], img.shape[1]
    if axis==0 :  
        if isnt_groundtruth :
            for i in range(0,lines):
                for j in range(0,columns):
                    for c in range(0, 3):
                        x[i,j,c]=img[lines-i-1,columns-j-1,c]
        else :

            for i in range(0,lines):
                for j in range(0,columns):
                    x[i,j]=img[lines-i-1,columns-j-1]
        
    elif axis==1 :
        if isnt_groundtruth:       

            for i in range(0,lines):
                for j in range(0,columns):
                    for c in range(0, 3):
                        x[i,j,c]=img[i,columns-j-1,c]
        else :

            for i in range(0,lines):
                for j in range(0,columns):
                    x[i,j]=img[i,columns-j-1]
        
    elif axis==2 :
        if isnt_groundtruth:

            for i in range(0,img.shape[0]):
                for j in range(0,img.shape[1]):
                    for c in range(0, img.shape[2]):
                        x[i,j,c]=img[lines-i-1,j,c]
        else :

            for i in range(0,img.shape[0]):
                for j in range(0,img.shape[1]):
                    x[i,j]=img[lines-i-1,j]
    return x


def create_rotat(filename, newfilename, isnt_groundtruth):
    '''Create a new image rotated with an angle of -pi/5 and saves it into the new file name.
    filename = file name of the original image
    newfilename = file name that will contain the rotated image
    isnt groundtruth = "The image is coded in RVB"
    '''
    img = load_image(filename)
    img_rotated = binary_to_uint8(rotation(img, isnt_groundtruth))
    Image.fromarray(img_rotated).save(newfilename)
    
def create_mirror(filename, newfilename, axis, isnt_groundtruth):
    '''Create a new image mirrored and saves it into the new file name.
    filename = file name of the original image
    newfilename = file name that will contain the mirrored image
    axis = 0 for central symmetry, 1 for horizontal symmetry, 2 for vertical symmetry
    isnt groundtruth = "The image is coded in RVB"
    '''
    img = load_image(filename)
    img_rotated = binary_to_uint8(mirror(img, axis, isnt_groundtruth))
    Image.fromarray(img_rotated).save(newfilename)

def increase_dataset(image_filenames, groundtruth_filenames, nb_aug):
    for i in range(nb_aug):
        print("Multiplying data stored in image n°"+str(i+1)+"...")
        # get the file names
        (filename, filename_rotat, filename_mirror, filename_rotat_mirror) = image_filenames[i]
        (filename_g, filename_rotat_g, filename_mirror_g, filename_rotat_mirror_g) = groundtruth_filenames[i]
        # simple rotations
        create_rotat(filename, filename_rotat, True)
        create_rotat(filename_g, filename_rotat_g, False)
        # choice of a random axis
        axis = rd.randint(0,2)
        create_mirror(filename, filename_mirror, axis, True)
        create_mirror(filename_g, filename_mirror_g, axis, False)
        # again for composition rotation + mirror
        axis = rd.randint(0,2)
        create_mirror(filename_rotat, filename_rotat_mirror, axis, True)
        create_mirror(filename_rotat_g, filename_rotat_mirror_g, axis, False)
    print("Done.")

if __name__ == '__main__':    
    beg = "training/images/satImage_"
    image_filenames = [(beg+"{0:03}.png".format(i+1),beg+"{0:03}.png".format(i+101),beg+"{0:03}.png".format(i+201),beg+"{0:03}.png".format(i+301)) for i in range(100)]
    beg = "training/groundtruth/satImage_"
    groundtruth_filenames =  [(beg+"{0:03}.png".format(i+1),beg+"{0:03}.png".format(i+101),beg+"{0:03}.png".format(i+201),beg+"{0:03}.png".format(i+301)) for i in range(100)]
    increase_dataset(image_filenames, groundtruth_filenames, 100)