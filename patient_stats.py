import shelve
import numpy as np
import pdb
import pandas as pd
import datetime as dt
import time
import sys
import os
from optparse import OptionParser

import util
util = reload(util)

def date_str_to_date(date_str):
	return dt.datetime.strptime(date_str, '%Y%m%d')

def date_to_date_str(date):
	return dt.datetime.strftime(date, '%Y%m%d')

def get_db_keys(stats):
	db_keys = np.unique([stat['db_key'] for stat in stats])
	return db_keys

def get_code_list(stat):

	if stat.has_key('code_list') == False:
		code_list = []
	else:
		if type(stat['code_list']) == type([]):
			code_list = stat['code_list']
		elif type(stat['code_list']) == type(''):
			code_list = util.read_list_files(stat['code_list'])
		else:
			raise ValueError('Unrecognized format for code_list')

	return code_list	

def get_dtype(stat):

	dtype = []
	dtype.append(('person','S10'))
	if (stat['calc'] in set(['first','last'])) == True:
		dtype.append((stat['name'], 'S10'))
	elif stat['calc'] in set(['count','range']):
		dtype.append((stat['name'], '<i4'))
	elif (stat['calc'] in set(['min','max','mean'])) == True:
		dtype.append((stat['name'], '<f8'))
	else:
		raise NotImplementedError('Calculation needs type specified')
		
	return dtype

def calc_stat(db, stat, people, verbose=True):

	assert (stat['calc'] in set(['first','last','range','count','min','max','mean'])), 'Unrecognized calculation'
	assert (stat['db_key'] in set(['icd9','loinc','loinc_vals','ndc','cpt','icd9_proc'])), 'Unrecognized database'

	is_df = (type(people) == type(pd.DataFrame()))
	code_list = get_code_list(stat)
	code_index_set = set([db.code_to_index[stat['db_key']][code] for code in code_list])

	if stat.get('calc_gfr', False) == True:
		db.load_db('loinc')

	rows = []
	start_run_time = time.time()
	est_run_time_at = 10000
	for i in range(len(people)):
		if verbose == True:
			if i % 10000 == 0:
				print i
			if i == est_run_time_at:
				est_run_time = (time.time() - start_run_time)*(float(len(people))/est_run_time_at)*(1/(60.))
				print 'Estimated run time (min): ' + str(round(est_run_time,2))


		if is_df == True:
			person = people['person'].iloc[i]
		else:
			person = people[i]

		# Get the person's data

		db_person = db.db[stat['db_key']][person]
		date_strs = db_person[0]
		X = db_person[1]

		if len(date_strs) == 0:
			nz = (np.array([]), np.array([]))
		else:
			nz = X.nonzero()

		nz_date_indices = nz[0]
		nz_code_indices = nz[1]

		# Initialize the lists that will hold the dates and values used in order to calculate the stat

		stat_date_strs = []
		stat_vals = []

		# Get the dates specific to the code lists provided or if no code list is provided get all the dates

		if len(code_list) != 0:

			for d, date_index in enumerate(nz_date_indices):
				code_index = nz_code_indices[d]
				# Is the code for a date in the code list?
				if (code_index in code_index_set) == True:
					date_str = date_strs[date_index]
					stat_date_strs.append(date_str)

					val = X[date_index, code_index]
					if stat.get('calc_gfr', False) == True:
						code = db.codes['loinc'][code_index]
						if code == '2160-0':
							age = people['age'].iloc[i]
							is_female = (people['gender'].iloc[i] == 'F')
							val = util.calc_gfr(val, age, is_female)
					stat_vals.append(val)
		else:

			stat_date_strs = [date_strs[date_index] for d, date_index in enumerate(nz_date_indices)]
			stat_vals = [X[date_index, nz_code_indices[d]] for d, date_index in enumerate(nz_date_indices)]

		assert len(stat_date_strs) == len(stat_vals)

		# Remove dates outside of a date range if columns for the date range for each patient are specified 

		if stat.has_key('date_range_cols') == True:

			if stat['date_range_cols'][0] == '':
				start_date = dt.datetime.min
			else:
				start_date = date_str_to_date(people[stat['date_range_cols'][0]].iloc[i])

			if stat['date_range_cols'][1] == '':
				end_date = dt.datetime.max
			else:
				end_date = date_str_to_date(people[stat['date_range_cols'][1]].iloc[i])

			stat_vals = [val for v, val in enumerate(stat_vals) if date_str_to_date(stat_date_strs[v]) >= start_date and date_str_to_date(stat_date_strs[v]) < end_date]
			stat_date_strs = [date_str for date_str in stat_date_strs if date_str_to_date(date_str) >= start_date and date_str_to_date(date_str) < end_date] 

		assert len(stat_date_strs) == len(stat_vals)

		# Remove observations more than a given # of days away from each other or without consecutive observations in a set of time period of a given # of days	
			
		if stat.has_key('consecutive_obs') == True:

			approach = stat['consecutive_obs'][0]
			assert (approach in set(['time_difference','time_period'])) == True

			if approach == 'time_difference':

				n_consecutive = stat['consecutive_obs'][1]
				threshold_days = stat['consecutive_obs'][2]
	
				assert n_consecutive >= 2
				
				date_strs_unique = np.unique(stat_date_strs)
				if len(date_strs_unique) >= n_consecutive:	
					new_date_strs = set()
					date_vec = np.array([dt.datetime.strptime(date_str, '%Y%m%d') for date_str in date_strs_unique])
					date_diff = np.array(map(lambda x: x.days, date_vec[1:len(date_vec)] - date_vec[0:(len(date_vec)-1)]))
					date_threshold = (date_diff < threshold_days)

					for d0 in range(len(date_threshold)-n_consecutive+2):
						is_consecutive = True
						for d1 in range(d0, d0+n_consecutive-1):
							if date_threshold[d1] == False:
								is_consecutive = False
								break
						if is_consecutive == True:
							new_date_strs.add(dt.datetime.strftime(date_vec[d0], '%Y%m%d'))

					stat_vals = [val for v, val in enumerate(stat_vals) if (stat_date_strs[v] in new_date_strs) == True]
					stat_date_strs = [date_str for date_str in stat_date_strs if (date_str in new_date_strs) == True] 
				else:
					stat_vals = []		
					stat_date_strs = []	
				
			elif approach == 'time_period': 

				n_consecutive = stat['consecutive_obs'][1]
				time_period_days = stat['consecutive_obs'][2]
				assert n_consecutive >= 2

				new_date_strs = set()
				date_strs_unique = np.unique(stat_date_strs)
				dates_unique = np.array(map(lambda x: dt.datetime.strptime(x, '%Y%m%d'), date_strs_unique))
					
				max_d = np.max(dates_unique)
				for d0 in range(len(dates_unique)):
					sd = dates_unique[d0]
					n_streak = 0 
					is_consecutive = False
					while sd < max_d:
						ed = sd + dt.timedelta(days=time_period_days)	
						n_obs = np.sum((dates_unique >= sd) & (dates_unique < ed))
						if n_obs == 0:
							break
						n_streak += 1
						if n_streak == n_consecutive:
							is_consecutive = True	
							break
						sd = ed
					if is_consecutive == True:
						new_date_strs.add(dt.datetime.strftime(dates_unique[d0], '%Y%m%d'))
				
				stat_vals = [val for v, val in enumerate(stat_vals) if (stat_date_strs[v] in new_date_strs) == True]
				stat_date_strs = [date_str for date_str in stat_date_strs if (date_str in new_date_strs) == True] 
		
		assert len(stat_date_strs) == len(stat_vals)
		
		# Remove values outside of a value range if columns for the value range for each patient are specified 

		if stat.has_key('value_range') == True:
			min_value = stat['value_range'][0]
			max_value = stat['value_range'][1]

			stat_date_strs = [date_str for d, date_str in enumerate(stat_date_strs) if stat_vals[d] >= min_value and stat_vals[d] < max_value]
			stat_vals = [val for v, val in enumerate(stat_vals) if val >= min_value and val < max_value]
		
		assert len(stat_date_strs) == len(stat_vals)

		# Only include the dates if there's at least two observations that are at least X days apart

		if stat.has_key('gap') == True:

			gap_threshold_days = stat['gap']
			assert type(gap_threshold_days) == type(1) 

			if len(stat_date_strs) > 0:
				stat_ds = np.array(map(lambda x: dt.datetime.strptime(x, '%Y%m%d'), stat_date_strs))
				max_stat_date = np.max(stat_ds)
				min_stat_date = np.min(stat_ds)
				diff = (max_stat_date - min_stat_date).days

				if diff < gap_threshold_days:	
					stat_date_strs = []
					stat_vals = []

		assert len(stat_date_strs) == len(stat_vals)

		# Calculate the stats

		if len(stat_date_strs) == 0:
			if stat.get('keep_missing', False) == True:
				row = [person]
				if (stat['calc'] in set(['first','last'])) == True:
					row.append('')
				elif (stat['calc'] in set(['min','max','mean'])) == True:
					row.append(np.nan)
				elif stat['calc'] in set(['count','range']):
					row.append(int(0))
				else:
					raise NotImplementedError('Calculation not implemented')
				rows.append(tuple(row))
		else:
			row = [person]
			if stat['calc'] == 'first':
				stat_dates = map(lambda x: date_str_to_date(x), stat_date_strs)
				x = date_to_date_str(np.min(stat_dates))
			elif stat['calc'] == 'last':
				stat_dates = map(lambda x: date_str_to_date(x), stat_date_strs)
				x = date_to_date_str(np.max(stat_dates))
			elif stat['calc'] == 'range':
				stat_dates = map(lambda x: date_str_to_date(x), stat_date_strs)
				x = (np.max(stat_dates) - np.min(stat_dates)).days
			elif stat['calc'] == 'count':
				# Count a maximum of one observation per day
				x = len(np.unique(stat_date_strs))
			elif stat['calc'] == 'min':
				x = np.min(stat_vals)
			elif stat['calc'] == 'max':
				x = np.max(stat_vals)	
			elif stat['calc'] == 'mean':
				x = np.mean(stat_vals)
			else:
				raise NotImplementedError('Calculation not implemented')

			row.append(x)
			rows.append(tuple(row))

	# Turn the data rows into a pandas DataFrame

	dtype = get_dtype(stat)
	data = pd.DataFrame.from_records(np.array(rows, dtype=dtype))

	# Add back in columns? 

	if stat.get('keep_cols', False) == True:
		assert is_df == True
		assert len(people['person'].unique()) == len(people['person'])
		nrows = len(data)
		data = pd.merge(data, people, on='person', how='left') 
		assert len(data) == nrows	

	return data

def drop_missing(fname):

	data = pd.read_csv(fname, sep='\t', dtype=str)
	assert len(data.columns) == 2
	assert data.columns[0] == 'person'
	data = data[(data[data.columns[1]] != '') & (data[data.columns[1]].isnull() == False)]
	return data

def drop_zero(fname):

	data = pd.read_csv(fname, sep='\t', dtype={0: 'S10', 1: '<i4'})
	assert len(data.columns) == 2
	assert data.columns[0] == 'person'
	data = data[(data[data.columns[1]] != 0) & (data[data.columns[1]].isnull() == False)]
	return data

def keep_threshold(fname, colname, threshold, comparison):

	data = pd.read_csv(fname, sep='\t', dtype={'person': 'S10', colname: '<f8'})
	if comparison == 'gt':
		data = data[(data[colname].isnull() == False) & (data[colname] > threshold)]
	elif comparison == 'ge':
		data = data[(data[colname].isnull() == False) & (data[colname] >= threshold)]
	elif comparison == 'lt':
		data = data[(data[colname].isnull() == False) & (data[colname] < threshold)]
	elif comparison == 'le':
		data = data[(data[colname].isnull() == False) & (data[colname] <= threshold)]
	elif comparison == 'eq':
		data = data[(data[colname].isnull() == False) & (data[colname] == threshold)]
	else:
		raise ValueError("Unrecognized comparison")
	
	return data

def merge(fname_a, fname_b):

	data_a = pd.read_csv(fname_a, sep='\t', dtype=str)
	data_b = pd.read_csv(fname_b, sep='\t', dtype=str)

	assert len(data_a.columns) == 2
	assert data_a.columns[0] == 'person'
	assert len(data_b.columns) == 2
	assert data_b.columns[0] == 'person'

	data = pd.merge(data_a, data_b, on='person')
	data = data[(data[data.columns[1]] != '') & (data[data.columns[1]].isnull() == False)]
	data = data[(data[data.columns[2]] != '') & (data[data.columns[2]].isnull() == False)]

	return data

def read_cohort(fname):

	data = pd.read_csv(fname, sep='\t', dtype={'person': str, 'y': int, 'start_date': str, 'end_date': str})
	assert len(data.columns) == 4	
	return data

def read_demographics(fname):
	data = pd.read_csv(fname, sep='\t', dtype={'person': str, 'age': int, 'gender': str})
	assert len(data.columns) == 3
	data = data[['person','age','gender']]
	return data

def sample(n_sample, db):
	np.random.seed(1)
	data = np.random.choice(db.people, n_sample, replace=False)
	return data

def get_people(stat, stats_key, db, out_dir):

	assert out_dir.endswith('/') == True or out_dir == ''

	if stat.has_key('input') == False:
		people = db.people
	else:
		if type(stat['input']) == type([]):
			func = stat['input'][0]
			args = stat['input'][1:len(stat['input'])]
		
			if func == 'drop_missing':
				fname = out_dir + stats_key + '_' + args[0] + '.txt'
				people = drop_missing(fname)
			elif func == 'drop_zero':
				fname = out_dir + stats_key + '_' + args[0] + '.txt'
				people = drop_zero(fname)
			elif func == 'merge':
				fname_a = out_dir + stats_key + '_' + args[0] + '.txt'
				fname_b = out_dir + stats_key + '_' + args[1] + '.txt'
				people = merge(fname_a, fname_b)
			elif func == 'sample':
				n_sample = args[0]
				people = sample(n_sample, db)
			elif func == 'read_cohort':
				fname = out_dir + stats_key + '_' + args[0] + '.txt'
				people = read_cohort(fname)
			elif func == 'read_demographics':
				fname = args[0]
				people = read_demographics(fname)
			elif func == 'keep_threshold':
				fname = out_dir + stats_key + '_' + args[0] + '.txt'	
				colname = args[0]
				threshold = float(args[1])
				comparison = args[2]
				people = keep_threshold(fname, colname, threshold, comparison)	
			else:	
				raise ValueError("Unrecognized function")
		elif type(stat['input']) == type(''):
			fname = out_dir + stats_key + '_' + stat['input'] + '.txt'
			with open(fname, 'r') as fin:
				line = fin.readline()
				if line.find('person') != -1:
					is_df = True
				else:
					is_df = False
			if is_df == True:
				people = pd.read_csv(fname, sep='\t', dtype=str)
			else:
				people = util.read_list_files(fname)

		else:
			raise ValueError("No method to parse input in this format")

	return people

def patient_stats(db, stats, stats_key, out_dir='', stat_indices=None, verbose=True, check_if_file_exists=False, save_files=True):

	if stat_indices is None:
		stat_indices = range(len(stats))

	data = {}
	for s in stat_indices:
		stat = stats[s]

		if verbose == True:
			print stat['name']

		people = get_people(stat, stats_key, db, out_dir)
		out_fname = out_dir + stats_key + '_' + stat['name'] + '.txt' 

		if check_if_file_exists == True and os.path.isfile(out_fname) == True:
				data[stat['name']] = pd.read_csv(out_fname, sep='\t')
		else:
			data[stat['name']] = calc_stat(db, stat, people, verbose)

			if save_files == True:
				data[stat['name']].to_csv(out_fname, sep='\t', index=False)

	return data

def main(data_paths_fname, stats_list_fname, stats_key, stat_indices=None, verbose=True, check_if_file_exists=False):

	data_paths = util.read_yaml(data_paths_fname)
	stats_list = util.read_yaml(stats_list_fname)
	stats = stats_list[stats_key]	
	db_keys = get_db_keys(stats)

	if stats_list.has_key('_config') == True:
		out_dir = stats_list['_config']['out_dir']
		assert out_dir.endswith('/')
	else:
		out_dir = ''

	db = util.Database(data_paths_fname)
	db.load_db(db_keys, people=True)

	start_time = time.time()
	data = patient_stats(db, stats, stats_key, out_dir, stat_indices, verbose, check_if_file_exists)
	if verbose == True:
		print time.time() - start_time

	return data

if __name__ == '__main__':

	desc = "Calculate patient stats, e.g. the number of lab observations each patient has for a given set of LOINC codes or the date of each patient's first ICD9 code"
	parser = OptionParser(description=desc, usage="usage: %prog [options] data_paths_fname stats_list_fname stats_key")
	parser.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False)	
	parser.add_option('-c', '--check_if_file_exists', action='store_true', dest='check_if_file_exists', default=False)	
	(options, args) = parser.parse_args()

	assert len(args) == 3 or len(args) == 4
		
	data_paths_fname = args[0]
	stats_list_fname = args[1]
	stats_key = args[2]
	if len(args) == 4:
		stat_indices = [int(args[3])]
	else:
		stat_indices = None

	main(data_paths_fname, stats_list_fname, stats_key, stat_indices, options.verbose, options.check_if_file_exists)
