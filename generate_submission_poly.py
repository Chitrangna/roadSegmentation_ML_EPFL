from helper import *
import os, re, csv
import numpy as np
import matplotlib.image as mpimg


def scipy_ridge(X,Y):
    from sklearn.linear_model import Ridge
    ridgemodel=Ridge(alpha=0.1)
    ridgemodel.fit(X,Y)
    return ridgemodel


def scipy_linear(X,Y):
    from sklearn import linear_model
    # we create an instance of the classifier and fit the data
    logreg = linear_model.LogisticRegression(C=1e5, max_iter=200, class_weight="balanced", solver='newton-cg')
    logreg.fit(X, Y)
    return logreg
   
    
def XY(patch_size, nb_imgs, extract_fct = extract_features_RVB_6d, features_needed=[]):
    # Load a set of images
    root_dir = "training/"
    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    n = min(nb_imgs, len(files)) # Load maximum 10 images
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
    
    gt_dir = root_dir + "groundtruth/"
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)] # get_imgs

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
    
    X = np.asarray([extract_fct(img_patches[i], grad_patches[i], sec_patches[i]) for i in range(len(img_patches))])
    Y = np.asarray([np.mean(gt_patches[i]) for i in range(len(gt_patches))])
    
    return X,Y




def mask_to_submission_strings(image_filename, patch_size, extract_fct, features_needed, weights, degree):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    Xi = extract_img_features(image_filename, extract_fct, features_needed)
    Zi = predict_poly(weights, Xi, degree) #Zi is a list of 0 and 1
    L=[]
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            id_patch = (j*(im.shape[1])//patch_size)//patch_size +i//patch_size
            # i, j =  abscissa and ordinate of the bottom-left corner of the patch we're predicting
            label = Zi[id_patch]
            a = "{:03d}_{}_{},{}\n".format(img_number, j, i, label)
            L.append(a)
    return L

def masks_to_submission(submission_filename, image_filenames, patch_size, extract_fct, features_needed, weights, degree):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,predictionnne\n')
        fieldnames = ['first_name', 'last_name', 'Grade']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        i=1
        for fn in image_filenames[0:]:
            print("Generating output for file "+str(i)+"/"+str(len(image_filenames))+"...")
            i+=1
            L =mask_to_submission_strings(fn, patch_size, extract_fct, features_needed, weights, degree)
            for l in L :
                f.write(l)

def true_submission_test_set():
    '''Returns the file names of the test set'''
    return ['test_set_images/test_' +  str(i) + '/test_' +  str(i) +'.png' for i in range(1,51)]

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def make_poly_predicter(Xtrain, Ytrain, degree=3):
    tx_train = build_poly(Xtrain, degree)
    weights = least_squares(Ytrain,tx_train)
    return weights

def predict_poly(weights, Xtest, degree):
    tx_test = build_poly(Xtest, degree)
    return tx_test.dot(weights)


def fcttostring(fct):
    '''Returns the name of a function object'''
    s = str(fct) 
    i = s.find(' ',10)
    return s[10:i]



def filter_threshold(filename, threshold):
    l = brut_to_list(filename)
    v_to_c = lambda x : int(x>threshold)
    with open(filename[:-4]+"_"+str(threshold)+".csv",'w') as filtered:
        wri = csv.writer(filtered, delimiter=',')
       
        wri.writerow(l[0])
        for row in l[1:]:
            wri.writerow([row[0],v_to_c(row[1])])
    filtered.close()    

    
def main(submission_filename = "prediction_poly.csv"):
    degree = 7
    patch_size =16
    nb_imgs_training = 10
    foreground_threshold = 0.21
    # extraction functions : 
        # extract_features_RVB_6d, extract_features_gray_2d, extract_features_all_8d
        # extract_features_grad_only, extract_features_grad_and_RVB, extract_features_grad_norm
        # extract_features_subpatch
    extract_fct = extract_features_RVB_6d
    extract_fct_name = fcttostring(extract_fct)
    features_needed = []
    print("Getting the training set...")
    codename = str(patch_size)+"_"+str(nb_imgs_training)+"_"+str(extract_fct_name)
    #submission_filename = 'poly_'+codename+'.csv'
    X,Y= XY(patch_size, nb_imgs_training, extract_fct, features_needed)
    
    print("Training set OK ! Training model... ")
    
    weights = make_poly_predicter(X, Y, degree)
    
    print("Model trained !")
    Z = ['test_set_images/test_' +  str(i) + '/test_' +  str(i) +'.png' for i in range(1,51)]
    masks_to_submission(submission_filename, Z, patch_size, extract_fct, features_needed, weights, degree)
    print("Finished !")

    return submission_filename
    
main()
