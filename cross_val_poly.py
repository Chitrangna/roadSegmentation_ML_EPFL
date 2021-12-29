from mask_to_submission import *
import os
import numpy as np
import matplotlib.image as mpimg
import re
from feature import *

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_crop(im, w, h):
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
    # img is a patch (shape 16, 16, 3)
    feat_gray = extract_features_2d(img, grad_img, sec_img)
    feat_color = extract_features_RVB_6d(img, grad_img, sec_img)
    feat = np.append(feat_gray,feat_color)
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
    img_patches = img_crop(img, patch_size, patch_size)
    if "grad" in features_needed :
        grads= compute_gradientT(img)
        grad_patches=  img_crop(grads, patch_size, patch_size)
    else : 
        grad_patches=  [0 for x in img_patches]
    if "sec" in features_needed :
        sec= compute_gradientT(img)
        sec_patches=  img_crop(sec, patch_size, patch_size)
    else : 
        sec_patches=  [0 for x in img_patches]
    X = np.asarray([extractfct(img_patches[i], grad_patches[i], sec_patches[i]) for i in range(len(img_patches))])
    return X




def scipy_mod(X,Y):
    from sklearn import linear_model
    # we create an instance of the classifier and fit the data
    logreg = linear_model.LogisticRegression(C=1e5, max_iter=200, class_weight="balanced", solver='newton-cg')
    logreg.fit(X, Y)
    return logreg


    
def XY_train(patch_size, filenames, extract_fct = extract_features_RVB_6d, features_needed=[], foreground_threshold=0.75):
    # Load a set of images
    root_dir = "training/"
    image_dir = root_dir + "images/"
    n=len(filenames)
    imgs = [load_image(image_dir + filename) for filename in filenames]
    
    gt_dir = root_dir + "groundtruth/"
    gt_imgs = [load_image(gt_dir + filename) for filename in filenames] # get_imgs

    # Extract patches from input images

    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]
 

    # Linearize list of patches, i is the index of the image
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    # len(img_patches) = 10 (nb of patches) * 25² (number of patches per image, 25 = 400/16)
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    
    if features_needed !=[] :  
        print("Computing additional features at pixel level...")
        if "grad" in features_needed :        
            grads= [compute_gradientT(img) for img in imgs]
            grad_patches = [img_crop(grads[i], patch_size, patch_size) for i in range(n)]    
            grad_patches = np.asarray([grad_patches[i][j] for i in range(len(grad_patches)) for j in range(len(grad_patches[i]))])
        else : grad_patches = [0 for x in img_patches]
        if "second" in features_needed :        
            sec= [compute_second_derivative(img) for img in imgs]
            sec_patches = [img_crop(sec[i], patch_size, patch_size) for i in range(n)]    
            sec_patches = np.asarray([sec_patches[i][j] for i in range(len(sec_patches)) for j in range(len(sec_patches[i]))])
        else : sec_patches = [0 for x in img_patches]
        print("Additional features computed !")
    if not "grad" in features_needed : 
        grad_patches = [0 for x in img_patches]
    if not "sec" in features_needed :
        sec_patches = [0 for x in img_patches]
        
    v_to_c = lambda v :int (np.sum(v) > foreground_threshold)
    X = np.asarray([extract_fct(img_patches[i], grad_patches[i], sec_patches[i]) for i in range(len(img_patches))])
    Y = np.asarray([v_to_c(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
    return X,Y




def mask_to_submission_strings_crossval(image_filename, model, patch_size, extract_fct, features_needed):
    """Reads a single image and outputs the strings that should go into the submission file"""
    image_filename = "training/images/"+image_filename
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    Xi = extract_img_features(image_filename, extract_fct, features_needed)
    Zi = model.predict(Xi)  #Zi is a list of 0 and 1
    return Zi
            

def masks_to_submission_crossval(submission_filename, image_filenames, model, patch_size, extract_fct, features_needed):
    """Converts images into a submission file"""
    dico = {}
    for filename in image_filenames:
        dico[filename] = mask_to_submission_strings_crossval(filename, model, patch_size, extract_fct, features_needed)
    return dico


def true_submission_test_set():
    return ['test_set_images/test_' +  str(i) + '/test_' +  str(i) +'.png' for i in range(1,51)]

def index_to_filename_training(x):
    return "satImage_{:03.0f}.png".format(int(x))

def groundtruth(filename, foreground_threshold):
    v_to_c = lambda v :int (np.sum(v) > foreground_threshold)

    gt_img = load_image('training/groundtruth/'+filename)
    gt_patches = img_crop(gt_img, patch_size, patch_size)
    gt_patches = list(map(np.mean, gt_patches))
    return list(map(v_to_c,gt_patches))
    

def run_test(training_set, test_set, extract_fct, features_needed, patch_size, foreground_threshold) :
    print("\n *** Running a new test ***")
    training_set,test_set = np.sort(training_set), np.sort(test_set)
    training_set = list(map(index_to_filename_training,training_set))
    test_set = list(map(index_to_filename_training,test_set))    
    print("Getting the training set...")
    X,Y= XY_train(patch_size, training_set, extract_fct, features_needed,foreground_threshold)
    print("Training set OK ! Training model... ")
    model = scipy_mod(X,Y)
    print("Model trained !")
    dico = masks_to_submission_crossval(submission_filename, test_set, model, patch_size, extract_fct, features_needed)
    
    # Now let's measure the performance
    true_pos, false_pos, true_neg, false_neg, weight = 0,0,0,0,0
    for filename in test_set :
        answer = groundtruth(filename, foreground_threshold)
        prediction = dico[filename]
        weight += len(prediction)
        for i in range(0,len(prediction)):
            if prediction[i]==1 :
                if answer[i]==1 :
                    true_pos+=1
                else :
                    false_pos+=1
            elif answer[i]==1 :
                false_neg+=1
            else : 
                true_neg+=1
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    print("Precision for given test set : ",precision)
    print("F-score for given test set : ", 2*precision*recall/(precision+recall))
    return (precision, recall, weight)
    
    

    
if __name__ == '__main__':
    submission_filename = 'crossval.csv'
    patch_size =4
    nb_imgs = 10
    nb_imgs_training = 8
    nb_imgs_test = nb_imgs-nb_imgs_training
    foreground_threshold = 0.15 # percentage of pixels > 1 required to assign a foreground label to a patch
                               # big threshold means lots of roads everywhere (precision bad)

    extract_fct = extract_features_grad_and_RVB
    features_needed = ["grad"]
    # extraction functions : 
        # extract_features_RVB_6d, extract_features_gray_2d, extract_features_all_8d
        # extract_features_grad_only, extract_features_grad_and_RVB, extract_features_grad_norm
    reorder = np.random.permutation(nb_imgs)+1
    :
    F, P, weight_tot = [],[],0
    #for i in range(0,nb_imgs//nb_imgs_test) 
    for i in range(0,3):
        # We chose set n°i (that is indexes in reorder[i,i+nb_imgs_test])
        beg = i*nb_imgs_test
        (precision, recall, weight)= run_test(
            reorder[beg:beg+nb_imgs_test], #training set
            np.append(reorder[:beg],reorder[beg+nb_imgs_test:]), #test set
            extract_fct, features_needed, 
            patch_size, foreground_threshold)
        F.append(2*precision*recall*weight/(precision+recall))
        P.append(precision*weight)
        weight_tot+=weight
    print("F1 - score : ", np.sum(F)/weight_tot)
    print("Precision : ", np.sum(P)/weight_tot)
    #return F, P
        
        

