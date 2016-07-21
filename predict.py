import numpy as np
import tables
import pdb
import pandas as pd
import yaml

import features
features = reload(features)

import emb
emb = reload(emb)

import models
models = reload(models)

def predict(in_fname, n_labs, age_index, gender_index, out_fname, verbose=False, emb_fnames=None):

	if verbose:
		print "loading data"

	X_train, Y_train, X_validation, Y_validation, X_test, Y_test = features.get_data(in_fname)

	emb_data_list = [None]
	if emb_fnames is not None:
		for emb_fname in emb_fnames:
			emb_data_list.append(emb.get_emb_data(emb_fname))
	else:
		emb_fnames = ['']

	if verbose:
		print "training, validating and testing models"

	results = []

	for e, emb_data in enumerate(emb_data_list):
		if verbose:
			print str(e)

		if verbose:
			print "-->L2"

		model = models.L2(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs, emb_data)
		model.crossvalidate(params=[[False, True], [0.01, 0.05, 0.1, 0.5, 1, 5, 10]], param_names=['fit_intercept', 'C'])
		model.test()
		s = model.summarize()
		s['emb_fname'] = emb_fnames[e]  
		results.append(s)

		if verbose:
			print "-->L1"

		model = models.L1(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs, age_index, gender_index, emb_data)
		model.crossvalidate(params=[[False, True], [0.01, 0.05, 0.1, 0.5, 1, 5, 10]], param_names=['fit_intercept', 'C'])
		model.test()
		s = model.summarize()
		s['emb_fname'] = emb_fnames[e]  
		results.append(s)

		'''
		if verbose:
			print "-->RandomForest"

		model = models.RandomForest(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, emb_data)
		params = [[1, 10, 20], [1, 3, 10], ['sqrt_n_features', 'n_features'], [1, 3, 10], [1, 3, 10], [True, False], ['gini', 'entropy']]
		param_names = ['n_estimators','max_depth','max_features','min_samples_split','min_samples_leaf','bootstrap','criterion']
		model.crossvalidate(params=params, param_names=param_names)
		model.test()
		s = model.summarize()
		s['emb_fname'] = emb_fnames[e]  
		results.append(s)
		'''

		if emb_data is not None:
			if verbose:
				print "-->Only embeddings"

			model = models.L(emb_data[0], Y_train, emb_data[1], Y_validation, emb_data[2], Y_test, None)
			model.crossvalidate(params=[['l1','l2'],[False, True], [0.01, 0.05, 0.1, 0.5, 1, 5, 10]], param_names=['penalty','fit_intercept','C'])
			model.test()
			s = model.summarize()
			s['emb_fname'] = emb_fnames[e]  
			results.append(s)

	with open(out_fname, 'w') as fout:
		fout.write(yaml.dump(results))

