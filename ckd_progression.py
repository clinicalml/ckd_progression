import pandas as pd
import numpy as np
import datetime as dt
from optparse import OptionParser
import pdb

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

def run(out_dir, config_fname, data_paths_fname, stats_list_fname, split_fname=None, check_if_file_exists=False, verbose=True): 

	data_paths = util.read_yaml(data_paths_fname)
	config = util.read_yaml(config_fname)

	stats_key = config['stats_key']
	outcome_stat_name = config['outcome_stat_name']
	cohort_stat_name = config.get('cohort_stat_name', None)
	lab_lower_bound = config.get('lab_lower_bound', None)
	lab_upper_bound = config.get('lab_upper_bound', None)
	gap_days = config.get('gap_days', None)
	training_window_days = config['training_window_days']
	buffer_window_days = config['buffer_window_days']
	outcome_window_days = config['outcome_window_days']
	time_period_days = config['time_period_days']
	time_scale_days = config['time_scale_days']
	use_just_labs = config['use_just_labs']
	feature_loincs_fname = config['feature_loincs_fname']
	add_age_sex = config['add_age_sex']
	calc_gfr = config['calc_gfr']
	regularizations = config.get('regularizations', [1])
	progression = config['progression']
	progression_lab_lower_bound = config.get('progression_lab_lower_bound', None)
	progression_lab_upper_bound = config.get('progression_lab_upper_bound', None)
	progression_gap_days = config.get('progression_gap_days', None)
	progression_stages = config.get('progression_stages', None)
	progression_init_stages = config.get('progression_init_stages', None)
	evaluate_nn = config.get('evaluate_nn', True)

	outcome_fname = out_dir + stats_key + '_' + outcome_stat_name + '.txt'
	if cohort_stat_name is None:
		cohort_fname = data_paths['demographics_fname']	
	else:
		cohort_fname = out_dir + stats_key + '_' + cohort_stat_name + '.txt'
	gfr_loincs = util.read_list_files('data/gfr_loincs.txt')
	training_data_fname = out_dir + stats_key + '_training_data.txt'

	feature_loincs = util.read_list_files(feature_loincs_fname)
	if use_just_labs == False:
		feature_diseases = [[icd9] for icd9 in util.read_list_files('data/kidney_disease_mi_icd9s.txt')]
		feature_drugs = [util.read_list_files('data/drug_class_'+dc.lower().replace('-','_').replace(',','_').replace(' ','_')+'_ndcs.txt') for dc in util.read_list_files('data/kidney_disease_drug_classes.txt')]
	else: 
		feature_diseases = []	
		feature_drugs = []

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
	if evaluate_nn:
		nn_predict_fname = out_dir + stats_key + '_nn_prediction_results.yaml'
	else:
		nn_predict_fname = None

	if verbose:
		print "Loading data"

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
		training_window_days, buffer_window_days, outcome_window_days, time_period_days, time_scale_days, gap_days, calc_gfr=True, verbose=verbose, \
		progression=progression, progression_lab_lower_bound=progression_lab_lower_bound, progression_lab_upper_bound=progression_lab_upper_bound, \
		progression_gap_days=progression_gap_days, progression_init_stages=progression_init_stages, progression_stages=progression_stages)
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

	predict.predict(features_split_fname, regularizations, n_labs, age_index, gender_index, predict_fname, nn_predict_fname)

if __name__ == '__main__':

	desc = "Run the full pipeline from cohort construction to prediction" 
	parser = OptionParser(description=desc, usage="usage: %prog [options] config_fname data_paths_fname stats_list_fname out_dir")
	parser.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False)
	parser.add_option('-c', '--check_if_file_exists', action='store_true', dest='check_if_file_exists', default=False)
	parser.add_option('-s', '--split_fname', action='store', dest='split_fname', default=None)
	(options, args) = parser.parse_args()

	assert len(args) == 4
	config_fname = args[0]
	data_paths_fname = args[1] 
	stats_list_fname = args[2]
	out_dir = args[3]

	run(out_dir, config_fname, data_paths_fname, stats_list_fname, split_fname=options.split_fname, check_if_file_exists=options.check_if_file_exists, verbose=options.verbose)
