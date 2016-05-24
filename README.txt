ckd_progression
===============

Predict end-stage kidney disease (defined as a kidney transplant or initiation of dialysis) in a cohort of patients with Stage IV chronic kidney disease based on relevant labs, diagnoses and prescriptions.

To run the full pipeline from cohort construction to prediction on the test data:

python ckd_progression.py tests/test_data_paths.yaml tests/test_stats.yaml tests/kidney_disease/

The program takes as input (1) a YAML file (e.g. tests/test_data_paths.yaml) with the paths to Python shelve databases storing the raw data and text files storing lists of patient IDs and medical codes, (2) a YAML file (e.g. tests/test_stats.yaml) that defines various statistics used to build a cohort and assign outcomes, and (3) a path to a directory in which to store the output.  

Tests
=====

To run tests from the command line:

nosetests
