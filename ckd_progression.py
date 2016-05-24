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

def create_outcome_data(out_dir):

	first_dialysis_cpt = pd.read_csv(out_dir + 'kidney_disease_first_dialysis_cpt.txt', sep='\t', dtype=str)
	first_kidney_transplant_cpt = pd.read_csv(out_dir + 'kidney_disease_first_kidney_transplant_cpt.txt', sep='\t', dtype=str)
	first_dialysis_icd9_proc = pd.read_csv(out_dir + 'kidney_disease_first_dialysis_icd9_proc.txt', sep='\t', dtype=str)
	first_kidney_transplant_icd9_proc = pd.read_csv(out_dir + 'kidney_disease_first_kidney_transplant_icd9_proc.txt', sep='\t', dtype=str)

	people = list(first_dialysis_cpt['person'].values)
	people += list(first_dialysis_icd9_proc['person'].values)
	people += list(first_kidney_transplant_cpt['person'].values)
	people += list(first_kidney_transplant_icd9_proc['person'].values)
	people = np.unique(people)

	data = pd.DataFrame({'person': people})
	data = pd.merge(data, first_dialysis_cpt, how='left', on='person')
	data = pd.merge(data, first_dialysis_icd9_proc, how='left', on='person')
	data = pd.merge(data, first_kidney_transplant_cpt, how='left', on='person')
	data = pd.merge(data, first_kidney_transplant_icd9_proc, how='left', on='person')

	data['first_kidney_failure'] = np.nan
	data = data.reset_index()
	for i in range(len(data)):
		ds = []
		if pd.isnull(data['first_kidney_transplant_cpt'].iloc[i]) == False:
			d = dt.datetime.strptime(data['first_kidney_transplant_cpt'].iloc[i], '%Y%m%d')
			ds.append(d)
		if pd.isnull(data['first_kidney_transplant_icd9_proc'].iloc[i]) == False:
			d = dt.datetime.strptime(data['first_kidney_transplant_icd9_proc'].iloc[i], '%Y%m%d')
			ds.append(d)
		if pd.isnull(data['first_dialysis_cpt'].iloc[i]) == False:
			d = dt.datetime.strptime(data['first_dialysis_cpt'].iloc[i], '%Y%m%d')
			ds.append(d)
		if pd.isnull(data['first_dialysis_icd9_proc'].iloc[i]) == False:
			d = dt.datetime.strptime(data['first_dialysis_icd9_proc'].iloc[i], '%Y%m%d')
			ds.append(d)

		data.loc[i, 'first_kidney_failure'] = dt.datetime.strftime(np.min(ds), '%Y%m%d')

	data = data[['person','first_kidney_failure']]
	data = data[data['first_kidney_failure'].isnull() == False]

	data.to_csv(out_dir + 'kidney_disease_first_kidney_failure.txt', index=False, sep='\t')

	return data

def run(out_dir, data_paths_fname, stats_list_fname, check_if_file_exists=False, verbose=True): 

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
	calc_gfr = True
	feature_loincs = util.read_list_files('data/ckd_loincs.txt')
	n_labs = len(feature_loincs)
	feature_diseases = [[icd9] for icd9 in util.read_list_files('data/kidney_disease_mi_icd9s.txt')]
	feature_drugs = [util.read_list_files('data/drug_class_'+dc.lower().replace('-','_').replace(',','_').replace(' ','_')+'_ndcs.txt') for dc in util.read_list_files('data/kidney_disease_drug_classes.txt')]	
	age_index = 45
	gender_index = 46
	features_fname = out_dir + stats_key + '_features.h5'
	split_fname = out_dir + stats_key + '_split.txt'
	features_split_fname = out_dir + stats_key + '_features_split.h5'
	predict_fname = out_dir + stats_key + '_prediction_results.yaml'

	# Load data

	data_paths = util.read_yaml(data_paths_fname)
	db = util.Database(data_paths_fname)
	db.load_people()
	db.load_db(['loinc','loinc_vals','cpt','icd9_proc','icd9','ndc'])

	stats = util.read_yaml(stats_list_fname)[stats_key]

	# Calc patient stats

	data = ps.patient_stats(db, stats, stats_key, out_dir, stat_indices=None, verbose=verbose, check_if_file_exists=check_if_file_exists, save_files=True)

	# Create outcome data

	data = create_outcome_data(out_dir)

	# Build training dataset

	cohort_data = btd.setup(data_paths['demographics_fname'], outcome_fname, cohort_fname)
	training_data = btd.build_training_data(db, cohort_data, gfr_loincs, lab_lower_bound, lab_upper_bound, \
		training_window_days, buffer_window_days, outcome_window_days, time_period_days, time_scale_days, gap_days, calc_gfr, verbose)
	training_data.to_csv(training_data_fname, index=False, sep='\t')

	# Build features

	features.features(db, training_data, feature_loincs, feature_diseases, feature_drugs, time_scale_days, features_fname, calc_gfr, verbose)
	features.train_validation_test_split(len(training_data['person'].unique()), split_fname)
	features.split(features_fname, features_split_fname, split_fname, verbose)
	
	# Train, test and validate models

	predict.predict(features_split_fname, n_labs, age_index, gender_index, predict_fname)

if __name__ == '__main__':

	desc = "Run the full pipeline from cohort construction to prediction" 
	parser = OptionParser(description=desc, usage="usage: %prog [options] data_paths_fname stats_list_fname out_dir")
	parser.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False)
	parser.add_option('-c', '--check_if_file_exists', action='store_true', dest='check_if_file_exists', default=False)
	(options, args) = parser.parse_args()

	assert len(args) == 3
	data_paths_fname = args[0] 
	stats_list_fname = args[1]
	out_dir = args[2]

	run(out_dir, data_paths_fname, stats_list_fname, check_if_file_exists=options.check_if_file_exists, verbose=options.verbose)
