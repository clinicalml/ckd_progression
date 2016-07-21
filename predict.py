import numpy as np
import tables
import pdb
import pandas as pd
import yaml

import features
features = reload(features)

import models
models = reload(models)

def predict(in_fname, n_labs, age_index, gender_index, out_fname, verbose=False):

	if verbose:
		print "loading data"

	X_train, Y_train, X_validation, Y_validation, X_test, Y_test = features.get_data(in_fname)

	if verbose:
		print "training, validating and testing models"

	results = []

	if verbose:
		print "-->L2"

	model = models.L2(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs)
	model.crossvalidate(params=[[False, True], [0.01, 0.05, 0.1, 0.5, 1, 5, 10]], param_names=['fit_intercept', 'C'])
	model.test()
	results.append(model.summarize())

	if verbose:
		print "-->L1"

	model = models.L1(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs, age_index, gender_index)
	model.crossvalidate(params=[[False, True], [0.01, 0.05, 0.1, 0.5, 1, 5, 10]], param_names=['fit_intercept', 'C'])
	model.test()
	results.append(model.summarize())

	if verbose:
		print "-->RandomForest"

	model = models.RandomForest(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
	params = [[1, 10, 20], [1, 3, 10], ['sqrt_n_features', 'n_features'], [1, 3, 10], [1, 3, 10], [True, False], ['gini', 'entropy']]
	param_names = ['n_estimators','max_depth','max_features','min_samples_split','min_samples_leaf','bootstrap','criterion']
	model.crossvalidate(params=params, param_names=param_names)
	model.test()
	results.append(model.summarize())

	with open(out_fname, 'w') as fout:
		fout.write(yaml.dump(results))
