import numpy as np
import pandas as pd
import datetime as dt
import cPickle as pickle
import scipy.sparse
import os
import pdb
import tables
import time

import util
util = reload(util)

np.random.seed(3)

def get_data(in_fname, get_person_ids=False):

	with tables.open_file(in_fname, mode='r') as fin:	
		X_train = fin.root.batch_input_train[:]
		Y_train = fin.root.batch_target_train[:]
		X_validation = fin.root.batch_input_validation[:]
		Y_validation = fin.root.batch_target_validation[:]
		X_test = fin.root.batch_input_test[:]
		Y_test = fin.root.batch_target_test[:]

		if get_person_ids:
			p_train = fin.root.p_train[:]
			p_validation = fin.root.p_validation[:]
			p_test = fin.root.p_test[:]

	if get_person_ids:
		return X_train, Y_train, X_validation, Y_validation, X_test, Y_test, p_train, p_validation, p_test
	else:
		return X_train, Y_train, X_validation, Y_validation, X_test, Y_test
	
def features(db, training_data, feature_loincs, feature_diseases, feature_drugs, time_scale_days, out_fname, calc_gfr=False, verbose=True, add_age_sex=False, \
	outcome_icd9s=[]):

	data = training_data.copy(deep=True)
	data = data.reset_index()

	if len(outcome_icd9s) == 0:
		multiple_outcomes = False
	else:
		multiple_outcomes = True
		outcome_icd9_indices = [db.code_to_index['icd9'][code] for code in outcome_icd9s]

	if multiple_outcomes:
		dcols = ['person','y','training_start_date','training_end_date','age','gender','outcome_start_date','outcome_end_date']
		ndcols = ['person','y','start_date','end_date','age','gender', 'outcome_start_date', 'outcome_end_date']
	else:
		dcols = ['person','y','training_start_date','training_end_date','age','gender']
		ndcols = ['person','y','start_date','end_date','age','gender']
 
	data = data[dcols]
	data.columns = ndcols
	data = data[ndcols]

	disease_loinc_indices = [db.code_to_index['loinc'][code] for code in feature_loincs]
	disease_icd9_indices = [set([db.code_to_index['icd9'][code] for code in codes]) for codes in feature_diseases] 
	drug_ndc_indices = [set([db.code_to_index['ndc'][code] for code in codes]) for codes in feature_drugs] 

	start_date = dt.datetime.strptime(data['start_date'].iloc[0], '%Y%m%d')
	end_date = dt.datetime.strptime(data['end_date'].iloc[0], '%Y%m%d')
	date_range = (end_date - start_date).days
	n_time = int(np.floor(date_range/float(time_scale_days)))
	n_features = len(feature_loincs) + len(feature_diseases) + len(feature_drugs)
	if add_age_sex:
		n_features += 2

	outlier_threshold = 3
	n_labs = len(feature_loincs)

	if multiple_outcomes:
		n_outcomes = len(outcome_icd9s)
	else:
		n_outcomes = 1

	# Open HDF5 file and initialize arrays

	with tables.open_file(out_fname, mode='w') as fout:
		X = fout.create_earray(fout.root, 'X', atom=tables.Atom.from_dtype(np.array([0.5]).dtype), shape=(0, 1, n_features, n_time))
		X_scaled = fout.create_earray(fout.root, 'X_scaled', atom=tables.Atom.from_dtype(np.array([0.5]).dtype), shape=(0, 1, n_features, n_time))
		Z = fout.create_earray(fout.root, 'Z', atom=tables.Atom.from_dtype(np.array([1]).dtype), shape=(0, 1, n_features, n_time))
		Y = fout.create_earray(fout.root, 'Y', atom=tables.Atom.from_dtype(np.array([1]).dtype), shape=(0, n_outcomes, 1, 1))
		P = fout.create_earray(fout.root, 'P', atom=tables.Atom.from_dtype(np.array([training_data['person'].iloc[0]]).dtype), shape=(0,))

		# Populate the arrays

		start_run_time = time.time()
		est_run_time_at = 5
		for i in range(len(data)):
			if verbose == True:
				if i % 100 == 0:
					print str(i)
				if i == est_run_time_at:
					est_run_time = (time.time() - start_run_time)*(float(len(data))/est_run_time_at)*(1/(60.))
					print 'Estimated run time (min): ' + str(round(est_run_time,2))

			# Get person specific data

			person = data['person'].iloc[i]
			start_date = dt.datetime.strptime(data['start_date'].iloc[i], '%Y%m%d')
			end_date = dt.datetime.strptime(data['end_date'].iloc[i], '%Y%m%d')
			y_person = int(data['y'].iloc[i])

			if multiple_outcomes:
				outcome_start_date = dt.datetime.strptime(data['outcome_start_date'].iloc[i], '%Y%m%d')
				outcome_end_date = dt.datetime.strptime(data['outcome_end_date'].iloc[i], '%Y%m%d')

			obs_date_strs = db.db['loinc'][person][0]
			obs_M = db.db['loinc'][person][1]
			val_M = db.db['loinc_vals'][person][1]

			icd9_date_strs = db.db['icd9'][person][0]
			icd9_M = db.db['icd9'][person][1]

			ndc_date_strs = db.db['ndc'][person][0]
			ndc_M = db.db['ndc'][person][1]

			age = int(data['age'].iloc[i])
			is_female = (data['gender'].iloc[i] == 'F')

			# Get lab values

			vals = {}
			for l, loinc_index in enumerate(disease_loinc_indices):
				for d, date_str in enumerate(obs_date_strs):
					date = dt.datetime.strptime(date_str, '%Y%m%d')
					if obs_M[d, loinc_index] == 1 and date >= start_date and date < end_date:
						t = int(np.floor(((date - start_date).days)/float(time_scale_days)))
						key = (l, t)
						if vals.has_key(key) == False:
							vals[key] = []	
	
						val = val_M[d, loinc_index]			
						if calc_gfr == True:
							code = db.codes['loinc'][loinc_index]
							if code == '2160-0':
								val = util.calc_gfr(val, age, is_female)

						if val > 0:
							vals[key].append(val)

			# Initialize arrays

			X_person = np.zeros((1, 1, n_features, n_time))
			Z_person = np.zeros((1, 1, n_features, n_time))
			Y_person = np.zeros((1, n_outcomes, 1, 1))

			# Aggregate lab values over time dimension

			for key in vals.keys():
				if len(vals[key]) > 0:
					X_person[0,0,key[0],key[1]] = np.mean(vals[key])
					Z_person[0,0,key[0],key[1]] = 1

			# Get icd9 values

			icd9_nz = icd9_M.nonzero()
			icd9_nz_date_indices = icd9_nz[0]
			icd9_nz_icd9_indices = icd9_nz[1] 

			for d, date_index in enumerate(icd9_nz_date_indices):
				icd9_index = icd9_nz_icd9_indices[d]
				disease_index = -1
				for c, indices_set in enumerate(disease_icd9_indices):
					if (icd9_index in indices_set) == True:
						disease_index = c
						break
					
				if disease_index != -1:
					date_str = icd9_date_strs[date_index]
					date = dt.datetime.strptime(date_str, '%Y%m%d')
					if date >= start_date and date < end_date:
						t = int(np.floor(((date - start_date).days)/float(time_scale_days)))
						X_person[0,0,disease_index + len(feature_loincs),t] = 1
						Z_person[0,0,disease_index + len(feature_loincs),t] = 1

			# Get icd9 values for outcome if multiple outcomes are being used

			if multiple_outcomes:

				for d, date_index in enumerate(icd9_nz_date_indices):
					icd9_index = icd9_nz_icd9_indices[d]
					disease_index = -1
					for c, o_idx in enumerate(outcome_icd9_indices):
						if o_idx == icd9_index:
							disease_index = c
							break
 					
					if disease_index != -1:
						date_str = icd9_date_strs[date_index]
						date = dt.datetime.strptime(date_str, '%Y%m%d')
						if date >= outcome_start_date and date < outcome_end_date:
							t = int(np.floor(((date - outcome_start_date).days)/float(time_scale_days)))
							Y_person[0,disease_index,0,0] = 1

			# Get ndc values

 			ndc_nz = ndc_M.nonzero()
			ndc_nz_date_indices = ndc_nz[0]
			ndc_nz_ndc_indices = ndc_nz[1] 

			for d, date_index in enumerate(ndc_nz_date_indices):
				ndc_index = ndc_nz_ndc_indices[d]
				drug_index = -1
				for c, indices_set in enumerate(drug_ndc_indices):
					if (ndc_index in indices_set) == True:
						drug_index = c
						break
					
				if drug_index != -1:
					date_str = ndc_date_strs[date_index]
					date = dt.datetime.strptime(date_str, '%Y%m%d')
					if date >= start_date and date < end_date:
						t = int(np.floor(((date - start_date).days)/float(time_scale_days)))
						X_person[0,0,drug_index + len(feature_loincs) + len(feature_diseases),t] = 1
						Z_person[0,0,drug_index + len(feature_loincs) + len(feature_diseases),t] = 1

			# Add age and sex

			if add_age_sex:
				age_index = len(feature_loincs) + len(feature_diseases) + len(feature_drugs)

				X_person[0,0,len(feature_loincs) + len(feature_diseases) + len(feature_drugs)] = age
				if is_female == True:
					X_person[0,0,len(feature_loincs) + len(feature_diseases) + len(feature_drugs) + 1] = 1.0
 
				Z_person[0,0,len(feature_loincs) + len(feature_diseases) + len(feature_drugs)] = 1	
				Z_person[0,0,len(feature_loincs) + len(feature_diseases) + len(feature_drugs) + 1] = 1

			# Add the person's data

			X.append(X_person)	
			Z.append(Z_person)

			if multiple_outcomes == False:
				Y_person[0,0,0,0] = y_person 
			Y.append(Y_person)

			P.append(np.array([person]))

		# Standardize and exclude outliers

		m = np.zeros(X.shape[2])
		s = np.ones(X.shape[2])
		for l in range(n_labs):
			x = X[:,0,l,:]
			x = x[x != 0]
			if len(x) >= 1:
				m[l] = np.mean(x)
				s[l] = np.std(x)
			if s[l] == 0:
				s[l] = 1.
	
		if add_age_sex:			
			x = X[:,0,age_index,:]
			x = x[x != 0]
			if len(x) >= 1:
				m[age_index] = np.mean(x)
				s[age_index] = np.std(x) 
			if s[age_index] == 0:
				s[age_index] = 1.

		X_scaled_vals = np.zeros(X.shape)
		for x0 in range(X.shape[0]):
			if x0 % 1000 == 0:
				if verbose == True:
					print x0
			for x1 in range(X.shape[1]):
				for x2 in range(X.shape[2]):	
					for x3 in range(X.shape[3]):
						if X[x0,x1,x2,x3] != 0:
							X_scaled_vals[x0,x1,x2,x3] = (X[x0,x1,x2,x3] - m[x2])/s[x2]
							if add_age_sex:
								if (np.abs(X_scaled_vals[x0,x1,x2,x3]) >= outlier_threshold) and (x2 != age_index):
									X_scaled_vals[x0,x1,x2,x3] = 0.
							else:
								if (np.abs(X_scaled_vals[x0,x1,x2,x3]) >= outlier_threshold):
									X_scaled_vals[x0,x1,x2,x3] = 0.

		X_scaled.append(X_scaled_vals)

		# Clean up

		fout.close()

def train_validation_test_split(people, out_fname, p_test=1./3, p_validation=1./3, prev_assignment_fname=None, prev_people=[], verbose=True):

	if prev_assignment_fname is not None:
		prev_assignment = util.read_list_files(prev_assignment_fname)		

	assignment = np.array(['none']*len(people), dtype='S20')
	for i in range(len(people)):
		if verbose:
			print i
		for j in range(len(prev_people)):
			if people[i] == prev_people[j]:
				assignment[i] = prev_assignment[j]
				break

	n_people = np.sum(assignment == 'none')
	n_test = int(p_test*n_people)
	n_validation = int(p_validation*n_people)
	n_train = n_people - n_test - n_validation
	assignment_remain = ['train']*n_train + ['validation']*n_validation + ['test']*n_test
	np.random.shuffle(assignment_remain)

	j = 0
	for i in range(len(people)):
		if assignment[i] == 'none':
			assignment[i] = assignment_remain[j]
			j += 1

	assert np.sum(assignment == 'none') == 0

	with open(out_fname, 'w') as fout:
		fout.write('\n'.join(assignment))

def split(in_fname, out_fname, assignment_fname, verbose=True):

	with tables.open_file(in_fname, mode='r') as fin:
		X = fin.root.X
		X_scaled = fin.root.X_scaled
		Z = fin.root.Z
		Y = fin.root.Y
		P = fin.root.P
		people = np.unique(P)
		person_to_index = dict((person, index) for index, person in enumerate(people))

		nrows = X.shape[0]
		n_outcomes = Y.shape[1]
		n_features = X.shape[2]
		n_time = X.shape[3]
		assignment = util.read_list_files(assignment_fname)

		shapes = {}
		dtypes = {}
		for split in ['train', 'validation', 'test']:	
			shapes['batch_input_'+split] = [1, n_features, n_time]
			dtypes['batch_input_'+split] = np.array([0.5]).dtype

			shapes['batch_input_nnx_'+split] = [1, n_features, n_time]
			dtypes['batch_input_nnx_'+split] = np.array([1]).dtype

			shapes['batch_target_'+split] = [n_outcomes, 1, 1]
			dtypes['batch_target_'+split] = np.array([1]).dtype

			shapes['p_'+split] = []
			dtypes['p_'+split] = np.array(['0123456789']).dtype

		with tables.open_file(out_fname, mode='w') as fout:

			arr = {}
			for key in shapes.keys():
				arr[key] = fout.create_earray(fout.root, key, atom=tables.Atom.from_dtype(dtypes[key]), shape=tuple([0] + shapes[key]))

			for i in range(nrows):
				if verbose:
					if i % 100 == 0:
						print i

				person = P[i]
				index = person_to_index[person]

				key = 'batch_input_'+assignment[index]
				arr[key].append(np.reshape(X_scaled[i,:,:,:], tuple([1] + shapes[key])))

				key = 'batch_input_nnx_'+assignment[index]
				arr[key].append(np.reshape(Z[i], tuple([1] + shapes[key])))

				key = 'batch_target_'+assignment[index]
				arr[key].append(np.reshape(Y[i], tuple([1] + shapes[key])))

				key = 'p_'+assignment[index]
				arr[key].append(np.reshape(P[i], tuple([1] + shapes[key])))
