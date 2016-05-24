import numpy as np
import tables
import pdb
import pandas as pd
import yaml

import models
models = reload(models)

def predict(in_fname, n_labs, age_index, gender_index, out_fname, verbose=False):

	if verbose:
		print "loading data"

	with tables.open_file(in_fname, mode='r') as fin:
		n_examples = fin.root.batch_input_train.nrows
		X_train = fin.root.batch_input_train[0:n_examples]
		Y_train = fin.root.batch_target_train[0:n_examples]
		X_validation = fin.root.batch_input_validation[0:n_examples]
		Y_validation = fin.root.batch_target_validation[0:n_examples]
		X_test = fin.root.batch_input_test[0:n_examples]
		Y_test = fin.root.batch_target_test[0:n_examples]

	if verbose:
		print "training, validating and testing models"

	results = []

	model = models.L2(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs)
	model.crossvalidate(params=[[False, True], [0.01, 0.1, 1, 5, 10, 20, 50, 100, 200]], param_names=['fit_intercept', 'C'])
	model.test()
	results.append(model.summarize())

	model = models.L1(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs, age_index, gender_index)
	model.crossvalidate(params=[[False, True], [0.01, 0.1, 1, 5, 10, 20, 50, 100, 200]], param_names=['fit_intercept', 'C'])
	model.test()
	results.append(model.summarize())

	model = models.RandomForest(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
	params = [[1, 10, 20], [1, 3, 10], [int(np.sqrt(X_train.shape[1])), X_train.shape[1]], [1, 3, 10], [1, 3, 10], [True, False], ['gini', 'entropy']]
	param_names = ['n_estimators','max_depth','max_features','min_samples_split','min_samples_leaf','bootstrap','criterion']
	model.crossvalidate(params=params, param_names=param_names)
	model.test()
	results.append(model.summarize())

	with open(out_fname, 'w') as fout:
		fout.write(yaml.dump(results))
