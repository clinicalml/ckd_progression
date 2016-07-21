import numpy as np
import pandas as pd
import datetime as dt
import sklearn.decomposition
import tables
import pdb
import h5py

import features
features = reload(features)

def reshape(X):
	return X.reshape((X.shape[0], np.prod(X.shape[1:])))

def emb(emb_features_fname, features_split_fname, emb_features_transformed_fname, verbose=True):

	batch_size = 256

	# reduce data

	model = sklearn.decomposition.IncrementalPCA(n_components=2) 
	
	with tables.open_file(emb_features_fname, 'r') as fin:
		nrows = fin.root.X_scaled.nrows
		X_scaled = fin.root.X_scaled

		batches = [list(batch) for batch in zip(range(0, nrows, batch_size), range(batch_size, nrows, batch_size))]

		for i, (start, stop) in enumerate(batches):
			if verbose:
				print str(i) + '/' + str(len(batches))

			X_batch = X_scaled[start:stop]
			model.partial_fit(reshape(X_batch))

	# transform data

	X_train, Y_train, X_validation, Y_validation, X_test, Y_test = features.get_data(features_split_fname)

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

