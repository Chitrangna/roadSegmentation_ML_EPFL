# ML project 2 : road segmentation

By Chitrangna Bhatt, Pierre  Liorit and Heloise Dupont de Dinechin, a.k.a HPC team for EPFL's 2019 machine learning course


## Installation

This project, fully coded in python, uses several usual libraries, including :
 - matplotlib
 - numpy, math, random
 - sklearn, tensorflow, 
 - os, sys, re, csv
 
## Usage

The main file run.py runs both polynomial and neural algorithms, printing information so that the reader knows what is happening.

## Code architecture

    - run.py makes calls to generate_submission_neural.py and generate_submission_poly.py, and then merges both predictions and filters the final result thanks to the code of post_processing.py.
   
    - generate_submission_poly.py makes prediction using polynomial regression. It stores images in .png files, and returns the list of paths to these results. The main function contains all the parameters of the regression, including :
        - degree of the regression
        - size of the training set
        - extraction function : do we want to use gradient coordinates, RGB code ?
    
    - generate_submission_neural.py makes prediction using a neural network with relu activation function. As the previous one, it writes prediction in files and returns the list of the file names. The most important parameters are 
        - The size of the training set, which should be less than 100 unless file feature_aug.py has been run
        - The number of epochs
    According to the values of this parameters, the running time can vary from 30 seconds to a dozen of hours.

    - cross_val_poly.py performs cross-validation for the polynomial algorithm. Its main parameters are the training size and validation size.
    
    - cross_val_neural.py performs cross-validation for the neural approach. Its main parameters are the training size and validation size. It should be a bit modified (using generate_submission_neural.py) to have rounded labels (real categories). For different thresholds, it computes precision, recall and F-score on a small validation test, for each threshold in a certain range.
        
    
    - helpers.py contains all useful functions, grouped in several categories :  
        - Functions to manipulate images : overlays, loading, concatenation
        - Functions to manipulate csv files : loading, writing
        - Functions manipulating submissions (playing with thresholds)
        - Mask to submission functions (from gray shades or black and white images to csv fsubmission file)
        - Submission to mask function (from labels or csv submission file to images, for better visualisation)
        - Image processing functions : compting gradient, second derivative...
        - Extraction functions that enable to extract particular features from the dataset
        
    - gaussianfilter.py can be used to apply a gaussian filter on images (training images, for example)



## Remarks

We used pieces of code provided by the ML course staff. In particular, files generate_submission_neural.py and 
We ran the codes on noto.epfl.ch


