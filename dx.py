import numpy as np
import pandas as pd
import pdb

import util
util = reload(util)

import features
features = reload(features)

def get_common_diseases(common_diseases_fname):

	common_diseases = []
	with open(common_diseases_fname, 'r') as fin:
		for line in fin:
			vals = line.split(':')
			part1_vals = vals[1].split('"')
			icd9 = part1_vals[1].split(' ')[0]
			common_diseases.append([icd9])

	return common_diseases

def dx(dx_features_fname, dx_features_split_fname, split_fname, feature_diseases, db, training_data_fname, time_scale_days, verbose=True):

	feature_loincs = []
	feature_drugs = []

	training_data = pd.read_csv(training_data_fname, sep='\t', dtype=str)

	# we want to relate the presence or absence of diagnoses in the outcome window the presence or absence of the label which is calculated based on codes in the outcome window
	training_data = training_data[['person','y','outcome_start_date','outcome_end_date','age','gender']]
	training_data.columns = ['person','y','training_start_date','training_end_date','age','gender']

	features.features(db, training_data, feature_loincs, feature_diseases, feature_drugs, time_scale_days, dx_features_fname, calc_gfr=False, verbose=verbose, add_age_sex=False)

	features.split(dx_features_fname, dx_features_split_fname, split_fname, verbose)
