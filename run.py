#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''File obtained to get the submission'''

from helper import *

submission_filename = "result.csv" 

print("Computing gray prediction from  neural approach saved...")

import generate_submission_neural
masks_neural = generate_submission_neural.main()
print("Turning masks into submissions...")
masks_to_submission("prediction_neural.csv", masks_neural)

print("Computing gray prediction from  polynomial approach saved in file.")

import generate_submission_poly
prediction_poly_filename = generate_submission_poly.main() 

print("Averageing both results...")

list_poly = brut_to_list(prediction_poly_filename)
list_neural = brut_to_list("prediction_neural.csv")

coeff=0.7   # Weight given to neural prediction. Polynomial prediciton is 1-coeff
make_average("prediction_poly.csv", "prediction_neural.csv", "average.csv", coeff)

print("Post-processing the average...")

filter_postprocessing("average.csv", submission_filename,"")

print("Submission saved in file "+submission_filename)



