import numpy as np
import pandas as pd
from optparse import OptionParser

import util
util = reload(util)

def create_and_load_database(data_paths_fname, db_key):
	database = util.Database(data_paths_fname)
	database.load_db(db_key, people=True)
	return database

def count(database, db_key, people=None, code_list=None, verbose=True):
	
	people_set = set(people)

	code_db = database.code_db[db_key]
	descs = database.descs.get(db_key, {})
	if code_list == None:
		code_list = database.codes[db_key]

	data = {db_key: [], 'desc': [], 'n_people': []}
	for i, code in enumerate(code_list):
		if verbose == True:
			print str(i) + ' of ' + str(len(code_list))

		data[db_key].append(code)
		data['desc'].append(descs.get(code, ''))

		if people is None:
			data['n_people'].append(len(code_db[code]))
		else:
			data['n_people'].append(len(code_db[code].intersection(people_set)))
	df = pd.DataFrame(data)
	df = df.sort('n_people', ascending=False)
	df = df[[db_key, 'desc', 'n_people']]

	return df

def main(data_paths_fname, db_key, code_list_fname, out_fname, verbose=True):

	if code_list_fname != None:
		code_list = util.read_list_files(code_list_fname)
	else:
		code_list = None

	database = create_and_load_database(data_paths_fname, db_key)
	df = count(database, db_key, code_list, verbose)
	df.to_csv(out_fname, sep='\t', index=False)	

	return df

if __name__ == '__main__':

	desc = 'Count the number of patients with each code for a given set of codes'
	parser = OptionParser(description=desc, usage="usage: %prog [options] data_paths_filename db_key out_fname")
	parser.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False)
	parser.add_option('-c', '--code_list', action='store', dest='code_list_fname', default=None)
	(options, args) = parser.parse_args()

	assert len(args) == 3

	data_paths_fname = args[0]
	db_key = args[1]
	out_fname = args[2]

	main(data_paths_fname, db_key, options.code_list_fname, out_fname, options.verbose)
