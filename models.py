import numpy as np
import sklearn.linear_model
import itertools
import pdb
import yaml 
import warnings

def evaluate(model, X, y):
	proba = model.predict_proba(X)[:,1]
	fpr, tpr, _ = sklearn.metrics.roc_curve(y, proba)
	
	if np.sum(np.isnan(tpr)) > 0:
		auc = -1
		warnings.warn("nans in true positive rate")
	else:
		auc = sklearn.metrics.auc(fpr, tpr)

	return auc

class Model():

	def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs):
		self.X_train, self.Y_train, self.labels = self.format_data(X_train, Y_train, n_labs)
		self.X_validation, self.Y_validation, _ = self.format_data(X_validation, Y_validation, n_labs)
		self.X_test, self.Y_test, _ = self.format_data(X_test, Y_test, n_labs)

		self.validation_auc = {}
		self.test_auc = None
		self.best_param = None
		self.best_auc = -np.inf

	def crossvalidate(self, params, param_names):

		self.params = params
		self.param_names = param_names
		self.param_name_to_index = dict((param_name, index) for index, param_name in enumerate(param_names))

		for param in itertools.product(*params):
			model = self.get_model(param)
			model.fit(self.X_train, self.Y_train)
			self.validation_auc[param] = evaluate(model, self.X_validation, self.Y_validation)

		for param in self.validation_auc.keys():
			if self.validation_auc[param] >= self.best_auc:
				self.best_auc = self.validation_auc[param]
				self.best_param = param

	def test(self):
		model = self.get_model(self.best_param)
		model.fit(self.X_train, self.Y_train)
		self.test_auc = evaluate(model, self.X_test, self.Y_test)

	def summarize(self):
		s = {'model': self.model, 'test_auc': float(self.test_auc), 'best_param': list(self.best_param), 'best_auc': float(self.best_auc), 'params': list(self.params), 'param_names': list(self.param_names)} 
		return s
	
class L2(Model):

	def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs):
		Model.__init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs)
		self.model = 'L2'

	def format_data(self, X, Y, n_labs):
		
		X_f = np.zeros((X.shape[0], X.shape[2]))

		for i in range(X.shape[0]):
			for l in range(n_labs):
				x = X[i,0,l,:]
				x = x[x != 0]

				if len(x) > 0:
					X_f[i,l] = np.mean(x)

			for d in range(n_labs, X.shape[2]):
				x = X[i,0,d,:]
				X_f[i,d] = np.max(x)

		y_f = Y[:,0,0,0]

		labels = []
		for l in range(n_labs):
			labels.append('mean_'+str(l))
		for d in range(n_labs, X.shape[2]):
			labels.append('max_'+str(d))

		return X_f, y_f, labels

	def get_model(self, param):
		return sklearn.linear_model.LogisticRegression(penalty='l2', C=param[self.param_name_to_index['C']], fit_intercept=param[self.param_name_to_index['fit_intercept']])
