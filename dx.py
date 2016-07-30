import numpy as np
import pandas as pd
import pdb
import datetime as dt
import sklearn.linear_model
import h5py

import util
util = reload(util)

import features
features = reload(features)

import models
models = reload(models)

def get_common_diseases(common_diseases_fname):

	common_diseases = []
	with open(common_diseases_fname, 'r') as fin:
		for line in fin:
			vals = line.split(':')
			part1_vals = vals[1].split('"')
			icd9 = part1_vals[1].split(' ')[0]
			common_diseases.append([icd9])

	return common_diseases

def get_XY(db, training_data_fname, verbose=False):
	
	training_data = pd.read_csv(training_data_fname, sep='\t', dtype=str)
	data = training_data

	X = np.zeros((len(data), len(db.codes['icd9'])))
	Y = np.zeros(len(data))
	exclude = np.zeros((len(data), len(db.codes['icd9'])))

	for i in range(len(data)):
		if verbose:
			if i % 1000 == 0:
				print str(i) + '/' + str(len(data))

		person = data['person'].iloc[i]

		training_start_date = dt.datetime.strptime(data['training_start_date'].iloc[i], '%Y%m%d')
		training_end_date = dt.datetime.strptime(data['training_end_date'].iloc[i], '%Y%m%d')

		outcome_start_date = dt.datetime.strptime(data['outcome_start_date'].iloc[i], '%Y%m%d')
		outcome_end_date = dt.datetime.strptime(data['outcome_end_date'].iloc[i], '%Y%m%d')

		Y[i] = data['y'].iloc[i]

		db_person = db.db['icd9'][person]
		date_strs = db_person[0]
		M = db_person[1]
		nz = M.nonzero()
		nz_date_indices = nz[0]
		nz_code_indices = nz[1]

		for j, code_index in enumerate(nz_code_indices):
			date_index = nz_date_indices[j]
			date_str = date_strs[date_index]
			date = dt.datetime.strptime(date_str, '%Y%m%d')

			if (date >= training_start_date) and (date < outcome_start_date):
				exclude[i,code_index] = 1

			if (date >= outcome_start_date) and (date < outcome_end_date):
				X[i,code_index] = 1
	
	return X, Y, exclude

def get_dx(db, X, Y, exclude, training_data_fname, split_fname, verbose=False):

	training_data = pd.read_csv(training_data_fname, sep='\t', dtype=str)
	p = training_data['person'].values
	assert len(p) == len(X)
	assert len(p) == len(Y)

	assignment = util.read_list_files(split_fname)
	p_unique = np.unique(p)
	assert len(p_unique) == len(assignment)
	
	X_map = {}
	Y_map = {}
	s = {}
	for dataset in ['train','validation','test']:
		if verbose:
			print dataset
		s[dataset] = set([person for i, person in enumerate(p_unique) if assignment[i] == dataset])
		include = np.array([(person in s[dataset]) for person in p]) 
		X_map[dataset] = X[include,:]
		Y_map[dataset] = np.zeros((np.sum(include), 1, 1, 1))
		Y_map[dataset][:,0,0,0] = Y[include]		

	model = models.L(X_map['train'], Y_map['train'], X_map['validation'], Y_map['validation'], X_map['test'], Y_map['test'])
	model.crossvalidate(params=[['l1'],[False, True], [0.01, 0.05, 0.1, 0.5, 1]], param_names=['penalty','fit_intercept','C'], verbose=verbose)
	model.test()
	summ = model.summarize()
	best_param = summ['best_param']

	l = model.get_model(best_param)
	l.fit(X_map['train'], Y_map['train'][:,0,0,0])

	indices = np.argsort(np.abs(l.coef_[0,:]))[::-1]
	codes = [db.codes['icd9'][index] for index in indices]
	descs = [db.descs.get('icd9', {}).get(code, '') for code in codes]
	N = np.sum(X, axis=0)[indices]

	summary = pd.DataFrame({'icd9': codes, 'desc': descs, 'n': N, 'coef': l.coef_[0,:][indices]})
	summary = summary[['icd9','desc','n','coef']]

	return summary, s

def create_tobe_excluded(db, outcome_icd9s, tobe_excluded_fname, s):

	outcome_indices = np.array([db.code_to_index['icd9'][code] for code in outcome_icd9s])

	fout = h5py.File(tobe_excluded_fname, 'w')	
	exclude_map = {}
	for dataset in ['train','validation','test']:
		if verbose:
			print dataset
		include = np.array([(person in s[dataset]) for person in p]) 
		exclude_map[dataset] = np.zeros((np.sum(include), len(outcome_icd9s)+1, 1, 1))
		for j in range(len(outcome_indices)):
			exclude_map[dataset][:,j+1,0,0] = exclude[include,outcome_indices[j]]
		fout.create_dataset('batch_tobe_excluded_outcomes_'+dataset, data=exclude_map[dataset])
	fout.close()
	
def dx(dx_features_fname, dx_features_split_fname, split_fname, feature_diseases, db, training_data_fname, time_scale_days, verbose=True):

	feature_loincs = []
	feature_drugs = []

	training_data = pd.read_csv(training_data_fname, sep='\t', dtype=str)

	# we want to relate the presence or absence of diagnoses in the outcome window the presence or absence of the label which is calculated based on codes in the outcome window
	training_data = training_data[['person','y','outcome_start_date','outcome_end_date','age','gender']]
	training_data.columns = ['person','y','training_start_date','training_end_date','age','gender']

	features.features(db, training_data, feature_loincs, feature_diseases, feature_drugs, time_scale_days, dx_features_fname, calc_gfr=False, verbose=verbose, add_age_sex=False)

	features.split(dx_features_fname, dx_features_split_fname, split_fname, verbose)
