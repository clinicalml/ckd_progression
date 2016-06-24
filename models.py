import numpy as np
import sklearn.linear_model
import sklearn.ensemble
import itertools
import pdb
import yaml 
import warnings

random_state = 3

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

	def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs=None, age_index=None, gender_index=None):
		self.X_train, self.Y_train, self.labels = self.format_data(X_train, Y_train, n_labs, age_index, gender_index)
		self.X_validation, self.Y_validation, _ = self.format_data(X_validation, Y_validation, n_labs, age_index, gender_index)
		self.X_test, self.Y_test, _ = self.format_data(X_test, Y_test, n_labs, age_index, gender_index)

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

	def format_data(self, X, Y, n_labs, age_index=None, gender_index=None):
	
		self.n_features = X.shape[2]	
		X_f = np.zeros((X.shape[0], self.n_features))

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
		return sklearn.linear_model.LogisticRegression(penalty='l2', C=param[self.param_name_to_index['C']], fit_intercept=param[self.param_name_to_index['fit_intercept']], random_state=random_state)

class L1(Model):

	def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs, age_index, gender_index):
		Model.__init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs, age_index, gender_index)
		self.model = 'L1'
		self.age_index = age_index
		self.gender_index = gender_index

	def format_data(self, X, Y, n_labs, age_index=None, gender_index=None):

		n_examples = X.shape[0]
		n_time = X.shape[3]
		features = []
		labels = []
		self.n_features = 0
		window_lens = [3, 6, 12]

		for window_len in window_lens:
			for l in range(n_labs):
				v = X[:,0,l,(n_time - window_len):n_time]
				inc = np.zeros(len(v))
				dec = np.zeros(len(v))
				fluc = np.zeros(len(v))
				m = np.zeros(len(v))
				for i in range(len(v)):
					u = v[i,:]
					u = u[u != 0]

					if len(u) >= 1:
						m[i] = np.mean(u)
		
					if len(u) >= 2:
						diff = u[:-1] - u[1:]
						if np.sum(diff > 0) > 0 and np.sum(diff < 0) > 0:
							fluc[i] = 1
	
						if (u[-1] - u[0]) > 0:
							inc[i] = 1

						if (u[-1] - u[0]) < 0:
							dec[i] = 1

				features.append(m)
				labels.append('mean_'+str(l)+'_over_'+str(window_len))
				self.n_features += 1

				features.append(inc)
				labels.append('inc_'+str(l)+'_over_'+str(window_len))
				self.n_features += 1

				features.append(dec)
				labels.append('dec_'+str(l)+'_over_'+str(window_len))
				self.n_features += 1

				features.append(fluc)
				labels.append('fluc_'+str(l)+'_over_'+str(window_len))
				self.n_features += 1
	
			for l in range(n_labs, X.shape[2]):
				v = X[:,0,l,(n_time - window_len):n_time]
				m = np.zeros(len(v))
				for i in range(len(v)):
					m[i] = np.max(v[i,:])

				if (age_index is not None) and (gender_index is not None):
					if ((l in set([age_index, gender_index])) == False) or (((l in set([age_index, gender_index])) == True) and window_len == 12):
						features.append(m)
						labels.append('max_'+str(l)+'_over_'+str(window_len))
						self.n_features += 1
				else:
					features.append(m)
					labels.append('max_'+str(l)+'_over_'+str(window_len))
					self.n_features += 1

		X_f = np.zeros((n_examples, self.n_features))
		for i in range(self.n_features):
			X_f[:,i] = features[i]

		y_f = Y[:,0,0,0]

		return X_f, y_f, labels

	def get_model(self, param):
		return sklearn.linear_model.LogisticRegression(penalty='l1', C=param[self.param_name_to_index['C']], fit_intercept=param[self.param_name_to_index['fit_intercept']], random_state=random_state)

class RandomForest(Model):

	def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test):
		Model.__init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
		self.model = 'RandomForest'

	def format_data(self, X, Y, n_labs=None, age_index=None, gender_index=None):
		self.n_features = np.prod(X.shape[1:])
		X_f = np.reshape(X, (X.shape[0], self.n_features))
		y_f = Y[:,0,0,0]
		labels = map(str, range(self.n_features))
		return X_f, y_f, labels

	def get_model(self, param):

		if self.param_name_to_index.has_key('n_estimators') == True:
			n_estimators = param[self.param_name_to_index['n_estimators']]
		else:
			n_estimators = 20

		if self.param_name_to_index.has_key('criterion') == True:
			criterion = param[self.param_name_to_index['criterion']]
		else:
			criterion = 'entropy'

		if self.param_name_to_index.has_key('max_depth') == True:
			max_depth = param[self.param_name_to_index['max_depth']]
		else:
			max_depth = 3

		if self.param_name_to_index.has_key('min_samples_split') == True:
			min_samples_split = param[self.param_name_to_index['min_samples_split']]
		else:
			min_samples_split = 1

		if self.param_name_to_index.has_key('min_samples_leaf') == True:
			min_samples_leaf = param[self.param_name_to_index['min_samples_leaf']]
		else:
			min_samples_leaf = 10

		if self.param_name_to_index.has_key('max_features') == True:
			if param[self.param_name_to_index['max_features']] == 'sqrt_n_features':
				max_features = int(np.sqrt(self.n_features))
			elif param[self.param_name_to_index['max_features']] == 'n_features':
				max_features = self.n_features
			else:
				raise ValueError("param value not recognized")
		else:
			max_features = self.n_features

		if self.param_name_to_index.has_key('bootstrap') == True:
			bootstrap = param[self.param_name_to_index['bootstrap']]
		else:
			bootstrap = True

		model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, \
			min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, bootstrap=bootstrap, random_state=random_state)
		return model
