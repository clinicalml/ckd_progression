import pandas as pd
import numpy as np
import datetime as dt
from optparse import OptionParser

import patient_stats as ps
ps = reload(ps)

import build_training_data as btd
btd = reload(btd)

import features
features = reload(features)

import predict
predict = reload(predict)

import util
util = reload(util)

def run(out_dir, data_paths_fname, stats_list_fname, use_just_common_labs=True, split_fname=None, check_if_file_exists=False, verbose=True): 

	stats_key = 'kidney_disease'
	outcome_stat_name = 'first_kidney_failure'
	cohort_stat_name = 'n_gap_stage45'
	outcome_fname = out_dir + stats_key + '_' + outcome_stat_name + '.txt'
	cohort_fname = out_dir + stats_key + '_' + cohort_stat_name + '.txt'
	gfr_loincs = util.read_list_files('data/gfr_loincs.txt')
	training_data_fname = out_dir + stats_key + '_training_data.txt'
	lab_lower_bound = 15
	lab_upper_bound = 30
	training_window_days = 12*30
	buffer_window_days = 3*30
	outcome_window_days = 12*30
	time_period_days = 4*30
	time_scale_days = 30
	gap_days = 90

	if use_just_common_labs == False:
		feature_loincs = util.read_list_files('data/ckd_loincs.txt')
		feature_diseases = [[icd9] for icd9 in util.read_list_files('data/kidney_disease_mi_icd9s.txt')]
		feature_drugs = [util.read_list_files('data/drug_class_'+dc.lower().replace('-','_').replace(',','_').replace(' ','_')+'_ndcs.txt') for dc in util.read_list_files('data/kidney_disease_drug_classes.txt')]
		add_age_sex = True
		calc_gfr = True
	else: 
		feature_loincs = util.read_list_files('data/common_loincs.txt')
		feature_diseases = []	
		feature_drugs = []
		add_age_sex = False
		calc_gfr = False

	n_labs = len(feature_loincs)

	if add_age_sex:
		age_index = len(feature_loincs) + len(feature_diseases) + len(feature_drugs)
		gender_index = len(feature_loincs) + len(feature_diseases) + len(feature_drugs) + 1
	else:
		age_index = None
		gender_index = None

	features_fname = out_dir + stats_key + '_features.h5'
	features_split_fname = out_dir + stats_key + '_features_split.h5'
	predict_fname = out_dir + stats_key + '_prediction_results.yaml'

	if verbose:
		print "Loading data"

	data_paths = util.read_yaml(data_paths_fname)
	db = util.Database(data_paths_fname)
	db.load_people()
	db.load_db(['loinc','loinc_vals','cpt','icd9_proc','icd9','ndc'])

	stats = util.read_yaml(stats_list_fname)[stats_key]

	if verbose:
		print "Calculating patient stats"

	data = ps.patient_stats(db, stats, stats_key, out_dir, stat_indices=None, verbose=verbose, check_if_file_exists=check_if_file_exists, save_files=True)

	if verbose:
		print "Building training data"

	outcome_data = btd.build_outcome_data(out_dir, outcome_fname)
	cohort_data = btd.setup(data_paths['demographics_fname'], outcome_fname, cohort_fname)
	# calc_gfr = True here because it's required to define the condition
	training_data = btd.build_training_data(db, cohort_data, gfr_loincs, lab_lower_bound, lab_upper_bound, \
		training_window_days, buffer_window_days, outcome_window_days, time_period_days, time_scale_days, gap_days, calc_gfr=True, verbose=verbose)
	training_data.to_csv(training_data_fname, index=False, sep='\t')

	if verbose:
		print "Building features"

	features.features(db, training_data, feature_loincs, feature_diseases, feature_drugs, time_scale_days, features_fname, calc_gfr, verbose, add_age_sex)

	if split_fname is None:
		split_fname = out_dir + stats_key + '_split.txt'
		features.train_validation_test_split(training_data['person'].unique(), split_fname, verbose=verbose)

	features.split(features_fname, features_split_fname, split_fname, verbose)
	
	if verbose:
		print "Training, validating and testing models"

	predict.predict(features_split_fname, n_labs, age_index, gender_index, predict_fname)

if __name__ == '__main__':

	desc = "Run the full pipeline from cohort construction to prediction" 
	parser = OptionParser(description=desc, usage="usage: %prog [options] data_paths_fname stats_list_fname out_dir")
	parser.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False)
	parser.add_option('-c', '--check_if_file_exists', action='store_true', dest='check_if_file_exists', default=False)
	parser.add_option('-s', '--split_fname', action='store', dest='split_fname', default=None)
	parser.add_option('-f', '--use_ckd_labs_and_non_lab_features', action='store_true', dest='use_ckd_labs_and_non_lab_features', default=False)
	(options, args) = parser.parse_args()

	assert len(args) == 3
	data_paths_fname = args[0] 
	stats_list_fname = args[1]
	out_dir = args[2]

	if options.use_ckd_labs_and_non_lab_features == True:
		use_just_common_labs = False
	else:
		use_just_common_labs = True

	run(out_dir, data_paths_fname, stats_list_fname, use_just_common_labs=use_just_common_labs, split_fname=options.split_fname, check_if_file_exists=options.check_if_file_exists, verbose=options.verbose)
