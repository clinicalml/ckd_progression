import numpy as np
import tables
import pdb
import pandas as pd
import yaml
import os

import features
features = reload(features)

import emb
emb = reload(emb)

import models
models = reload(models)

import nn
nn = reload(nn)

def analyze(in_fname):

	data = yaml.load(open(in_fname, 'r'))

	results = {}	
	results['model'] = []
	results['test_auc'] = []
	results['validate_auc'] = []
	results['use_emb'] = []
	results['emb_fname'] = []

	for i in range(len(data)):
		results['model'].append(data[i]['model'])
		results['test_auc'].append(data[i]['test_auc'])
		results['validate_auc'].append(data[i]['best_auc'])
		results['use_emb'].append(data[i]['use_emb'])
		results['emb_fname'].append(data[i]['emb_fname'].split('/')[-1])

	results = pd.DataFrame(results)
	results = results.sort('test_auc', ascending=False)

	return results

def predict(in_fname, lin_n_cv_iters, n_cv_iters, regularizations, n_labs, age_index, gender_index, out_fname, nn_out_fname=None, verbose=False, emb_fnames=None):

	if verbose:
		print "loading data"

	X_train, Y_train, X_validation, Y_validation, X_test, Y_test = features.get_data(in_fname)

	emb_data_list = [None]
	emb_fname_list = ['']
	if emb_fnames is not None:
		for emb_fname in emb_fnames:
			emb_data_list.append(emb.get_emb_data(emb_fname))
			emb_fname_list.append(emb_fname)

	if verbose:
		print "training, validating and testing models"

	results = []

	for e, emb_data in enumerate(emb_data_list):
		if verbose:
			print str(e)

		if verbose:
			print "-->L2"

		model = models.L2(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs, emb_data)
		if lin_n_cv_iters == -1:
			params = [[False, True], regularizations]
		else:
			params = [['sample',False,True],['uniform',regularizations[0],regularizations[-1]]]
		
		model.crossvalidate(params=params, param_names=['fit_intercept', 'C'], n_cv_iters=lin_n_cv_iters)
		model.test()
		s = model.summarize()
		s['emb_fname'] = emb_fname_list[e] 
		results.append(s)

		if verbose:
			print "-->L1"

		model = models.L1(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs, age_index, gender_index, emb_data)
		if lin_n_cv_iters == -1:
			params = [[False, True], regularizations]
		else:
			params = [['sample',False,True],['uniform',regularizations[0],regularizations[-1]]]
		model.crossvalidate(params=params, param_names=['fit_intercept', 'C'], n_cv_iters=lin_n_cv_iters)
		model.test()
		s = model.summarize()
		s['emb_fname'] = emb_fname_list[e]  
		results.append(s)

		if verbose:
			print "-->RandomForest"

		model = models.RandomForest(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, emb_data)
		if n_cv_iters == -1:
			params = [[1, 10, 20], [1, 3, 10], ['sqrt_n_features', 'n_features'], [1, 3, 10], [1, 3, 10], [True, False], ['gini', 'entropy']]
		else:
			params = [['randint',1,20],['randint',1,10],['sample','sqrt_n_features','n_features'],['randint',1,10],['randint',1,10],['sample',True,False],['sample','gini','entropy']]
		param_names = ['n_estimators','max_depth','max_features','min_samples_split','min_samples_leaf','bootstrap','criterion']
		model.crossvalidate(params=params, param_names=param_names, n_cv_iters=n_cv_iters)
		model.test()
		s = model.summarize()
		s['emb_fname'] = emb_fname_list[e]  
		results.append(s)

		if emb_data is not None:
			if verbose:
				print "-->Only embeddings"

			model = models.L(emb_data[0], Y_train, emb_data[1], Y_validation, emb_data[2], Y_test, None)
			if lin_n_cv_iters == -1:
				params = [['l1','l2'],[False, True], regularizations]
			else:
				params = [['sample','l1','l2'],['sample',False,True],['uniform',regularizations[0],regularizations[-1]]]

			model.crossvalidate(params=params, param_names=['penalty','fit_intercept','C'], n_cv_iters=lin_n_cv_iters)
			model.test()
			s = model.summarize()
			s['emb_fname'] = emb_fname_list[e]  
			results.append(s)

	with open(out_fname, 'w') as fout:
		fout.write(yaml.dump(results))

	if nn_out_fname is not None:
		best_model = nn.evaluate(nn_out_fname, n_cv_iters, 20, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, 45, random_seed=345, verbose=verbose)

