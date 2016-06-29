import tables
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
import pdb
import lasagne
import sklearn.metrics
import yaml
import random

def evaluate(results_fname, X_train, Y_train, X_validation, Y_validation, X_test, Y_test):

	random_seed = 345
	n_epochs = 32

	best_valid_auc = -np.inf
	best_model = None

	np.random.seed(random_seed)

	for i in range(100):
		print i

		n_hidden = np.random.randint(10, 200)
		n_filters = np.random.randint(4, 32)
		k_horiz = np.random.randint(2, 10)
		dropout = np.random.uniform(0, 0.75)
		init_learning_rate = np.random.uniform(0.001, 2)
		rho = np.random.uniform(0.5, 0.95)	

		model = NeuralNet(n_hidden, n_filters, k_horiz, dropout, init_learning_rate, rho, random_seed)
		model.train_and_validate(n_epochs, X_train, Y_train, X_validation, Y_validation)
		if model.best_valid_auc > best_valid_auc:
			best_valid_auc = model.best_valid_auc
			best_model = model

	test_auc = best_model.calc_auc(X_test, Y_test)
	
	results = {}
	results['valid_auc'] = float(best_model.best_valid_auc)
	results['test_auc'] = float(test_auc)	
	results['config'] = best_model.config
	
	with open(results_fname, 'w') as fout:
		yaml.dump(results, fout)

def get_data(in_fname):

	with tables.open_file(in_fname, mode='r') as fin:
		n_examples = fin.root.batch_input_train.nrows
		X_train = fin.root.batch_input_train[0:n_examples]
		Y_train = fin.root.batch_target_train[0:n_examples]
		X_validation = fin.root.batch_input_validation[0:n_examples]
		Y_validation = fin.root.batch_target_validation[0:n_examples]
		X_test = fin.root.batch_input_test[0:n_examples]
		Y_test = fin.root.batch_target_test[0:n_examples]

	return X_train, Y_train, X_validation, Y_validation, X_test, Y_test

class NeuralNet():

	def __init__(self, n_hidden, n_filters, k_horiz, dropout, init_learning_rate, rho, random_seed):

		n_classes = 2
		n_features = 15
		n_time = 12
		pool_horiz = 3

		self.config = {}
		self.config['n_hidden'] = n_hidden	
		self.config['n_filters'] = n_filters
		self.config['k_horiz'] = k_horiz
		self.config['dropout'] = dropout
		self.config['init_learning_rate'] = init_learning_rate
		self.config['rho'] = rho
		self.config['random_seed'] = random_seed

		self.random_seed = random_seed
		self.rng = random.Random(x=self.random_seed)
		self.srng = theano.tensor.shared_randomstreams.RandomStreams(seed=random_seed)

		X = T.tensor4()
		Y = T.itensor4()
		
		tdim1 = 1 + n_time - k_horiz
		tdim2 = 1 + tdim1 - pool_horiz
		n_conv = int(n_filters*n_features*tdim2)

		w1 = theano.shared(value=self.init_weights((n_filters, 1, 1, k_horiz)))
		w2 = theano.shared(value=self.init_weights((n_conv, n_hidden)))
		b2 = theano.shared(value=np.zeros(n_hidden, dtype=theano.config.floatX))
		w3 = theano.shared(value=self.init_weights((n_hidden, n_classes)))
		b3 = theano.shared(value=np.zeros(n_classes, dtype=theano.config.floatX))

		log_prob = self.get_log_prob(X, w1, w2, b2, w3, b3, pool_horiz, n_conv, dropout, deterministic=False)
		test_log_prob = self.get_log_prob(X, w1, w2, b2, w3, b3, pool_horiz, n_conv, dropout, deterministic=True)

		self.params = [w1, w2, b2, w3, b3]

		y = Y[:,0,0,0]
		prediction = T.argmax(log_prob, axis=1)
		train_loss = -T.mean(log_prob[T.arange(X.shape[0]), y])

		updates = lasagne.updates.adadelta(train_loss, self.params, learning_rate=init_learning_rate, rho=rho, epsilon=1e-06) 

		self.train_fn = theano.function(inputs=[X, Y], outputs=train_loss, updates=updates, allow_input_downcast=True)
		self.predict_fn = theano.function(inputs=[X], outputs=test_log_prob, allow_input_downcast=True) 

	def init_weights(self, shape):
		n = np.prod(shape)
		x = np.array([self.rng.gauss(0, 1) for i in range(n)]) * 0.01
		return np.asarray(x.reshape(shape), dtype=theano.config.floatX)

	def add_dropout(self, l, dropout, deterministic):
		if dropout > 0:
			if deterministic:
				l = dropout*l
			else:
				l = T.switch(self.srng.binomial(size=l.shape, p=(1. - dropout)), l, 0)
		return l

	def get_log_prob(self, X, w1, w2, b2, w3, b3, pool_horiz, n_conv, dropout, deterministic):

		l1 = T.nnet.relu(T.nnet.conv2d(X, w1, border_mode='valid', subsample=(1, 1)))
		l2 = max_pool_2d(l1, ds=(1, pool_horiz), st=(1, 1), ignore_border=True)
		l3 = l2.reshape((X.shape[0], n_conv))	
		l3 = self.add_dropout(l3, dropout, deterministic)
		l4 = T.nnet.relu(T.dot(l3, w2) + b2)
		l4 = self.add_dropout(l4, dropout, deterministic)
		l5 = T.dot(l4, w3) + b3
		log_prob = T.nnet.logsoftmax(l5)

		return log_prob 

	def calc_auc(self, X, Y):
		proba = self.predict_fn(X)[:,1]
		fpr, tpr, _ = sklearn.metrics.roc_curve(Y[:,0,0,0], proba)
		auc = sklearn.metrics.auc(fpr, tpr)	
		return auc

	def load_params(self, new_params):
		for p in range(len(self.params)):
			self.params[p].set_value(new_params[p])
	
	def train_and_validate(self, n_epochs, X_train, Y_train, X_validation, Y_validation):

		self.rng = random.Random(x=self.random_seed)
	
		mini_batch_size = 256
		n_examples = X_train.shape[0]

		self.best_valid_auc = -np.inf
		self.best_valid_epoch = None
		self.best_params = None
		for epoch in range(n_epochs):
			indices = np.arange(n_examples)
			self.rng.shuffle(indices)
			X = X_train[indices]
			Y = Y_train[indices]

			for start, stop in zip(range(0, n_examples, mini_batch_size), range(mini_batch_size, n_examples, mini_batch_size)):
				train_loss = self.train_fn(X[start:stop], Y[start:stop])

			train_auc = self.calc_auc(X_train, Y_train)
			valid_auc = self.calc_auc(X_validation, Y_validation)
	
			if valid_auc > self.best_valid_auc:
				self.best_valid_auc = valid_auc
				self.best_valid_epoch = epoch
				self.best_params = [param.get_value() for param in self.params]

		self.load_params(self.best_params)	
