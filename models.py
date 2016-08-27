import numpy as np
import sklearn.linear_model
import sklearn.ensemble
import itertools
import pdb
import yaml 
import warnings

import emb
emb = reload(emb)

def add_feature_names_to_labels(labels, db, feature_loincs, feature_diseases, feature_drug_classes):

	feature_loinc_names = map(lambda x: db.descs['loinc'][x], feature_loincs)
	feature_disease_names = map(lambda x: db.descs['icd9'][x], feature_diseases)
	feature_drug_names = feature_drug_classes

	feature_names = feature_loinc_names + feature_disease_names + feature_drug_names + ['age','sex']

	feature_map = dict((feature_index, feature_name) for feature_index, feature_name in enumerate(feature_names))
	new_labels = []
	for label in labels:
		vals = label.split('_')
		new_label = vals[0] + '_' + feature_map[int(vals[1])] + '_' + vals[2] + '_' + vals[3]
		new_labels.append(new_label)

	return new_labels 

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

	def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs=None, age_index=None, gender_index=None, emb_data=None, verbose=False):
		self.X_train, self.Y_train, self.labels = self.format_data(X_train, Y_train, n_labs, age_index, gender_index, verbose=verbose)
		self.X_validation, self.Y_validation, _ = self.format_data(X_validation, Y_validation, n_labs, age_index, gender_index, verbose=verbose)
		self.X_test, self.Y_test, _ = self.format_data(X_test, Y_test, n_labs, age_index, gender_index, verbose=verbose)

		if emb_data is not None:
			self.X_train = emb.add_emb(self.X_train, emb_data[0])
			self.X_validation = emb.add_emb(self.X_validation, emb_data[1])
			self.X_test = emb.add_emb(self.X_test, emb_data[2])

			assert self.X_train.shape[1] == self.X_validation.shape[1]
			assert self.X_validation.shape[1] == self.X_test.shape[1]
			self.n_features = self.X_train.shape[1]
		
			self.use_emb = True
		else:
			self.use_emb = False

		self.validation_auc = {}
		self.test_auc = None
		self.best_auc = -np.inf
		self.best_param = None

		self.test_auc_std = -np.inf
		self.best_auc_std = -np.inf

		self.test_auc_perc = None
		self.best_auc_perc = None

		self.test_auc_map = {}
		self.best_auc_map = {}
		self.best_param_map = {}

		self.perc = [0,1,5,10,25,50,75,90,95,99,100]

	def crossvalidate(self, params, param_names, n_cv_iters=-1, verbose=False):

		self.params = params
		self.param_names = list(param_names) + ['random_state']
		self.param_name_to_index = dict((param_name, index) for index, param_name in enumerate(self.param_names))

		if n_cv_iters != -1:
			random_states = range(20)
		else:
			random_states = [3]

		if n_cv_iters != -1:

			for random_state in random_states:

				np.random.seed(345)

				for cv_iter in range(n_cv_iters):

					par = []
					for p in params:				
						if p[0] == 'uniform':
							v = np.random.uniform(p[1], p[2])
						elif p[0] == 'randint':
							v = np.random.randint(p[1], p[2]+1)
						elif p[0] == 'sample':
							arr = np.array(p[1:])
							idx = np.argmax(np.random.random(len(arr)))
							v = arr[idx]
						else:
							raise ValueError("unrecognized option")
						par.append(v)
		
					param = [p for p in par]
					param.append(random_state)	
					param = tuple(param)

					model = self.get_model(param)
					model.fit(self.X_train, self.Y_train)
					self.validation_auc[param] = evaluate(model, self.X_validation, self.Y_validation)
			
		else:

			for par in itertools.product(*params):
				if verbose:
					print param

				for random_state in random_states: 
					param = tuple(list(par) + [random_state])
					model = self.get_model(param)
					model.fit(self.X_train, self.Y_train)
					self.validation_auc[param] = evaluate(model, self.X_validation, self.Y_validation)

		for param in self.validation_auc.keys():	
			key = param[self.param_name_to_index['random_state']]
			if self.validation_auc[param] >= self.best_auc_map.get(key, -1):
				self.best_auc_map[key] = self.validation_auc[param]
				self.best_param_map[key] = param

		self.best_auc = np.mean(self.best_auc_map.values())
		self.best_auc_std = np.std(self.best_auc_map.values())
		self.best_auc_perc = np.percentile(self.best_auc_map.values(), self.perc).tolist()

	def test(self):
		for key in self.best_param_map.keys():
			model = self.get_model(self.best_param_map[key])
			model.fit(self.X_train, self.Y_train)
			self.test_auc_map[key] = evaluate(model, self.X_test, self.Y_test)

		self.test_auc = np.mean(self.test_auc_map.values())
		self.test_auc_std = np.std(self.test_auc_map.values())
		self.test_auc_perc = np.percentile(self.test_auc_map.values(), self.perc).tolist()

	def convert(self, field):
		if type(field) == np.string_:
			return str(field)
		elif type(field) == np.bool_:
			return bool(field)
		elif type(field) == np.int_:
			return int(field)
		elif type(field) == np.float_:
			return float(field)
		else:
			return field 

	def summarize(self):

		params = [list(param) for param in self.params]	
		s = {'model': self.model, 'params': params, 'param_names': list(self.param_names)}
		s['test_auc'] = float(self.test_auc)
		s['best_auc'] = float(self.best_auc)
		s['use_emb'] = self.use_emb	
		s['n_features'] = int(self.n_features) 

		s['best_auc_std'] = float(self.best_auc_std)
		s['test_auc_std'] = float(self.test_auc_std)
	
		s['best_auc_perc'] = self.best_auc_perc
		s['test_auc_perc'] = self.test_auc_perc
	
		s['best_param_map'] = []
		s['best_auc_map'] = []
		s['keys'] = [self.convert(key) for key in np.sort(self.best_param_map.keys())]
		for key in s['keys']:
			s['best_param_map'].append([self.convert(p) for p in self.best_param_map[key]])
			s['best_auc_map'].append(self.convert(self.best_auc_map[key]))

		s['perc'] = self.perc
	
		return s

class L(Model):

	def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, emb_data=None):
		Model.__init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, emb_data=emb_data)
		self.model = 'L'

	def format_data(self, X, Y, n_labs=None, age_index=None, gender_index=None, verbose=False):	
		self.n_features = int(np.prod(X.shape[1:]))
		X_f = X.reshape((X.shape[0], self.n_features))
		labels = map(str, range(self.n_features))
		return X_f, Y[:,0,0,0], labels

	def get_model(self, param):
		return sklearn.linear_model.LogisticRegression(penalty=param[self.param_name_to_index['penalty']], \
			C=param[self.param_name_to_index['C']], fit_intercept=param[self.param_name_to_index['fit_intercept']], random_state=param[self.param_name_to_index['random_state']])

class LMax(Model):

	def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, emb_data=None):
		Model.__init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, emb_data=emb_data)
		self.model = 'LMax'

	def format_data(self, X, Y, n_labs=None, age_index=None, gender_index=None, verbose=False):
	
		X_f = np.max(X[:, 0, :, :], axis=2)
		self.n_features = X_f.shape[1]
	
		y_f = Y[:,0,0,0]

		labels = []
		for i in range(self.n_features):
			labels.append('max_'+str(i))

		return X_f, y_f, labels

	def get_model(self, param):
		return sklearn.linear_model.LogisticRegression(penalty='l1', C=param[self.param_name_to_index['C']], fit_intercept=param[self.param_name_to_index['fit_intercept']], random_state=param[self.param_name_to_index['random_state']])
	
class L2(Model):

	def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs, emb_data=None, verbose=False):
		Model.__init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs, emb_data=emb_data, verbose=verbose)
		self.model = 'L2'

	def format_data(self, X, Y, n_labs, age_index=None, gender_index=None, verbose=False):
	
		self.n_features = X.shape[2]	
		X_f = np.zeros((X.shape[0], self.n_features))

		for i in range(X.shape[0]):
			if verbose:
				print str(i) + '/' + str(X.shape[0])

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
		return sklearn.linear_model.LogisticRegression(penalty='l2', C=param[self.param_name_to_index['C']], fit_intercept=param[self.param_name_to_index['fit_intercept']], random_state=param[self.param_name_to_index['random_state']])

class L1(Model):

	def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs, age_index, gender_index, emb_data=None, verbose=False):
		Model.__init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, n_labs, age_index, gender_index, emb_data=emb_data, verbose=verbose)
		self.model = 'L1'
		self.age_index = age_index
		self.gender_index = gender_index

	def format_data(self, X, Y, n_labs, age_index=None, gender_index=None, verbose=False):

		n_examples = X.shape[0]
		n_time = X.shape[3]
		features = []
		labels = []
		self.n_features = 0
		window_lens = [3, 6, 12]

		for window_len in window_lens:
			if verbose:
				print str(window_len)
			for l in range(n_labs):
				if verbose:
					print "--->"+str(l) + '/' + str(n_labs)

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
		return sklearn.linear_model.LogisticRegression(penalty='l1', C=param[self.param_name_to_index['C']], fit_intercept=param[self.param_name_to_index['fit_intercept']], random_state=param[self.param_name_to_index['random_state']])

class RandomForest(Model):

	def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, emb_data=None):
		Model.__init__(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, emb_data=emb_data)
		self.model = 'RandomForest'

	def format_data(self, X, Y, n_labs=None, age_index=None, gender_index=None, verbose=False):
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
			min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, bootstrap=bootstrap, random_state=param[self.param_name_to_index['random_state']])
		return model
