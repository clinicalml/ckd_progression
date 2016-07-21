import numpy as np
import pandas as pd
import datetime as dt
import sklearn.decomposition
import tables
import pdb
import h5py
import os

import features
features = reload(features)

def add_R(X, R):
	assert X.shape[0] == R.shape[0]	
	new_X = np.zeros((X.shape[0], X.shape[1] + R.shape[1]))
	new_X[:,:X.shape[1]] = X
	new_X[:,X.shape[1]:] = R
	return new_X

def get_reps(in_fname):
	
	with tables.open_file(in_fname, mode='r') as fin:
		R_train = fin.root.R_train[:]
		R_validation = fin.root.R_validation[:]
		R_test = fin.root.R_test[:]

	return R_train, R_validation, R_test

def reshape4(R):
	X = np.zeros((R.shape[0], 1, 1, R.shape[1]))
	X[:,0,0,:] = R
	return X

def reshape(X):
	return X.reshape((X.shape[0], np.prod(X.shape[1:])))

def emb(emb_features_path, features_split_path, emb_features_transformed_fname, verbose=True):

	if os.path.isdir(emb_features_path):
		fnames = os.listdir(emb_features_path)
	else:
		fnames = [emb_features_path]

	batch_size = 256

	# reduce data

	model = sklearn.decomposition.IncrementalPCA(n_components=200) 
	
	for fno, fname in enumerate(fnames):
		if verbose:
			print str(fno) + '/' + str(len(fnames))

		if os.path.isdir(emb_features_path):
			path = os.path.join(emb_features_path, fname)
		else:
			path = fname

		with tables.open_file(path, 'r') as fin:
			nrows = fin.root.X_scaled.nrows
			X_scaled = fin.root.X_scaled

			batches = [list(batch) for batch in zip(range(0, nrows, batch_size), range(batch_size, nrows, batch_size))]

			for i, (start, stop) in enumerate(batches):
				if verbose:
					print "-->" + str(i) + '/' + str(len(batches))

				X_batch = X_scaled[start:stop]
				model.partial_fit(reshape(X_batch))

	# transform data

	if os.path.isdir(features_split_path):

		datasets = ['train','validation','test']
		X_list = {}
		n = {}
		for dataset in datasets:
			X_list[dataset] = []
			n[dataset] = 0

		fnames = os.listdir(features_split_path)
		for fno, fname in enumerate(fnames):
			print "-->" + str(fno) + '/' + str(len(fnames))

			path = os.path.join(features_split_path, fname)
			with tables.open_file(path, 'r') as fin:
				X_scaled = fin.root.X_scaled[:]

			for dataset in datasets:
				if fname.find(dataset) != -1:
					X_list[dataset].append(X_scaled)
					n[dataset] += X_scaled.shape[0]
					break

		X = {}
		for dataset in datasets:
			print dataset

			X[dataset] = np.zeros((n[dataset], X_scaled.shape[1]))
			start = 0
			stop = 0
			for b in range(len(X_list[dataset])):
				print "-->" + str(b) + '/' + str(len(X_list[dataset]))
				stop += X_list[dataset][b].shape[0]
				X[dataset][start:stop] = X_list[dataset][b]
				start = stop

		X_train = X['train']
		X_validation = X['validation']
		X_test = X['test']
			 
	else:
		X_train, Y_train, X_validation, Y_validation, X_test, Y_test = features.get_data(features_split_path)

	R_train = model.transform(reshape(X_train))
	R_validation = model.transform(reshape(X_validation))
	R_test = model.transform(reshape(X_test))

	# write output

	fout = h5py.File(emb_features_transformed_fname, 'w')
	fout.create_dataset('R_train', data=R_train)
	fout.create_dataset('R_validation', data=R_validation)
	fout.create_dataset('R_test', data=R_test)
	fout.close()

def emb_features(db, feature_loincs, features_split_fname, training_window_days, time_scale_days, emb_features_fname, verbose=True):

	p_sample = 0.1
	colnames = ['age', 'gender', 'person', 'training_end_date', 'training_start_date', 'y']

	X_train, Y_train, X_validation, Y_validation, X_test, Y_test, p_train, p_validation, p_test = features.get_data(features_split_fname, True)
	p_exclude = set(list(p_validation) + list(p_test))

	people = np.array([person for person in db.people if (person in p_exclude) == False])
	r = np.random.rand(len(people))
	indices = np.argsort(r)
	n_sample = int(p_sample*len(people)) 
	sample_people = people[indices[0:n_sample]]

	training_data = dict((colname, []) for colname in colnames)

	for i, person in enumerate(sample_people):
		if verbose:
			print str(i) + '/' + str(len(sample_people))

		date_strs = db.db['loinc'][person][0]
		
		if len(date_strs) > 0:

			dates = map(lambda x: dt.datetime.strptime(x, '%Y%m%d'), date_strs)

			r = np.random.rand(len(dates))
			indices = np.argsort(r)

			sd = dates[indices[0]]

			training_start_date = dates[indices[0]]
			training_end_date = training_start_date + dt.timedelta(days=training_window_days)

			training_data['person'].append(person)
			training_data['age'].append(-1)
			training_data['gender'].append(-1)
			training_data['y'].append(-1)
			training_data['training_start_date'].append(dt.datetime.strftime(training_start_date, '%Y%m%d'))
			training_data['training_end_date'].append(dt.datetime.strftime(training_end_date, '%Y%m%d'))	

	training_data = pd.DataFrame(training_data)
	
	feature_diseases = []
	feature_drugs = []
	calc_gfr = False
	add_age_sex = False

	features.features(db, training_data, feature_loincs, feature_diseases, feature_drugs, time_scale_days, emb_features_fname, calc_gfr=calc_gfr, verbose=verbose, add_age_sex=add_age_sex)

