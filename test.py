import shelve
import numpy as np
import pandas as pd
import scipy.sparse
import os
import tables
import pdb

import util
util = reload(util)

import ckd_progression as ckd
ckd = reload(ckd)

tests_dir = 'tests/'
soln_dir = tests_dir + 'soln/'

if os.path.exists(tests_dir) == False:
	os.mkdir(tests_dir)

if os.path.exists(soln_dir) == False:
	os.mkdir(soln_dir)

def add_person(db, codes, person, dates, data, code_indices, date_indices):

	X = scipy.sparse.csr_matrix((data, (date_indices, code_indices)), shape=(len(dates), len(codes)), dtype=np.float64)
	db[person] = (dates, X)

	return db

def create_demographics(people, tests_dir):

	ages = [20,40,80,80,80,80,80,80,80,80]
	genders = ['M','F','M','M','M','M','M','M','M','M']

	data = {}
	data['person'] = []
	data['age'] = []
	data['gender'] = []
	demographics_fname = tests_dir + 'test_demographics.txt'
	with open(demographics_fname, 'w') as fout:
		for i in range(len(people)):
			data['person'].append(people[i])
			data['age'].append(ages[i])
			data['gender'].append(genders[i])
	data = pd.DataFrame(data)
	data[['age','gender','person']].to_csv(tests_dir + 'test_demographics.txt', index=False, sep='\t')

	return data

def create_db():
	
	cpt_db_fname = tests_dir + 'test_cpt_person_to_code.db'
	loinc_db_fname = tests_dir + 'test_loinc_person_to_code.db'
	loinc_vals_db_fname = tests_dir + 'test_loinc_vals_person_to_code.db'
	cpt_list_fname = tests_dir + 'test_cpt_list.txt'
	icd9_proc_list_fname = tests_dir + 'test_icd9_proc_list.txt'
	icd9_proc_db_fname = tests_dir + 'test_icd9_proc_person_to_code.db'
	loinc_list_fname = tests_dir + 'test_loinc_list.txt'
	people_list_fname = tests_dir + 'test_people_list.txt'
	icd9_list_fname = tests_dir + 'test_icd9_list.txt'
	icd9_db_fname = tests_dir + 'test_icd9_person_to_code.db'
	ndc_list_fname = tests_dir + 'test_ndc_list.txt'
	ndc_db_fname = tests_dir + 'test_ndc_person_to_code.db'

	cpts = util.read_list_files(cpt_list_fname)
	loincs = util.read_list_files(loinc_list_fname)
	people = util.read_list_files(people_list_fname)
	icd9_procs = util.read_list_files(icd9_proc_list_fname)
	icd9s = util.read_list_files(icd9_list_fname)
	ndcs = util.read_list_files(ndc_list_fname)

	loinc_db = shelve.open(loinc_db_fname)
	loinc_vals_db = shelve.open(loinc_vals_db_fname)
	cpt_db = shelve.open(cpt_db_fname)
	icd9_proc_db = shelve.open(icd9_proc_db_fname)
	icd9_db = shelve.open(icd9_db_fname)
	ndc_db = shelve.open(ndc_db_fname)

	for person in people:
		cpt_db[person] = (np.array([], dtype=object), scipy.sparse.csr_matrix(([], ([],[])), dtype=np.bool, shape=(0, len(cpts))))		
		loinc_vals_db[person] = (np.array([], dtype=object), scipy.sparse.csr_matrix(([], ([],[])), dtype=np.float64, shape=(0, len(loincs))))		
		loinc_db[person] = (np.array([], dtype=object), scipy.sparse.csr_matrix(([], ([],[])), dtype=np.bool, shape=(0, len(loincs))))
		icd9_proc_db[person] = (np.array([], dtype=object), scipy.sparse.csr_matrix(([], ([],[])), dtype=np.bool, shape=(0, len(icd9_procs))))
		icd9_db[person] = (np.array([], dtype=object), scipy.sparse.csr_matrix(([], ([],[])), dtype=np.bool, shape=(0, len(icd9s))))
		ndc_db[person] = (np.array([], dtype=object), scipy.sparse.csr_matrix(([], ([],[])), dtype=np.bool, shape=(0, len(ndcs))))

	# 437 = 01990 (kidney transplant), 5779 = 50380 (kidney transplant), 5 = 00099 (not a kidney transplant)
	cpt_db = add_person(cpt_db, cpts, people[0], np.array(['20110102','20100101','20121015'], dtype=object), [1,1,1], [437,5779,5], [1,0,2])
	cpt_db = add_person(cpt_db, cpts, people[2], np.array(['20110601'], dtype=object), [1], [437], [0])
	cpt_db = add_person(cpt_db, cpts, people[3], np.array(['20110601'], dtype=object), [1], [437], [0])
	cpt_db = add_person(cpt_db, cpts, people[4], np.array(['20110601'], dtype=object), [1], [437], [0])
	cpt_db = add_person(cpt_db, cpts, people[5], np.array(['20110601'], dtype=object), [1], [437], [0])
	pd.DataFrame({'person': [people[0], people[2], people[3], people[4], people[5]], 'first_kidney_transplant_cpt': ['20100101'] + ['20110601']*4}).to_csv(soln_dir + 'first_kidney_transplant_cpt.txt', index=False, sep='\t')

	# 1182 = 3942 (dialysis)
	icd9_proc_db = add_person(icd9_proc_db, icd9_procs, people[0], np.array(['20090504', '20090401'], dtype=object), [1,1], [1182, 1182], [0, 1]) 
	pd.DataFrame({'person': [people[0]], 'first_dialysis_icd9_proc': ['20090401']}).to_csv(soln_dir + 'first_dialysis_icd9_proc.txt', index=False, sep='\t')

	pd.DataFrame({'person': [], 'first_dialysis_cpt': []}).to_csv(soln_dir + 'first_dialysis_cpt.txt', index=False, sep='\t')
	pd.DataFrame({'person': [], 'first_kidney_transplant_icd9_proc': []}).to_csv(soln_dir + 'first_kidney_transplant_icd9_proc.txt', index=False, sep='\t')
	pd.DataFrame({'person': [people[0], people[2], people[3], people[4], people[5]], 'first_kidney_failure': ['20090401'] + ['20110601']*4}).to_csv(soln_dir + 'first_kidney_failure.txt', index=False, sep='\t')

	# 3225 = 33914-3 (GFR), 4026 = 48642-3 (GFR), 4027 = 48643-1 (GFR), 1909 = 2160-0 (Creatinine)
	loinc_db = add_person(loinc_db, loincs, people[0], np.array(['20100101','20110101'], dtype=object), [1, 1, 1], [3225,4026,4027], [0, 1, 1])
	loinc_vals_db = add_person(loinc_vals_db, loincs, people[0], np.array(['20100101','20110101'], dtype=object), [30, 16, 40], [3225,4026,4027], [0, 1, 1])

	for person_index in range(1, 10):
		loinc_db = add_person(loinc_db, loincs, people[person_index], np.array(['20100101','20100501', '20100901', '20101101'], dtype=object), [1, 1, 1, 1], [3225, 4026, 4026, 4026], [0, 1, 2, 3])
		loinc_vals_db = add_person(loinc_vals_db, loincs, people[person_index], np.array(['20100101','20100501', '20100901', '20101101'], dtype=object), [25, 18, 20, 22], [3225, 4026, 4026, 4026], [0, 1, 2, 3])

	d = {'person': people, 'min_gfr': [16.0] + [18.0]*9, 'age': [20, 40] + [80]*8, 'gender': ['M', 'F'] + ['M']*8}
	pd.DataFrame(d).to_csv(soln_dir + 'min_gfr.txt', index=False, sep='\t')
	pd.DataFrame({'person': people[1:10], 'n_gap_stage45': [4]*9}).to_csv(soln_dir + 'n_gap_stage45.txt', index=False, sep='\t')

	td = {'person': people[1:10], 'training_start_date': ['20100101']*9, 'training_end_date': ['20101227']*9, \
		'outcome_start_date': ['20110327']*9, 'outcome_end_date': ['20120321']*9, 'y': [0, 1, 1, 1, 1, 0, 0, 0, 0], 'age': [40] + [80]*8, 'gender': ['F'] + ['M']*8}
	pd.DataFrame(td).to_csv(soln_dir + 'training_data.txt', index=False, sep='\t')

	n_features = 47
	n_time = 12
	n_outcomes = 1
	age_index = 45
	gender_index = 46
	with tables.open_file(soln_dir + 'features.h5', mode='w') as fout:
		X = fout.create_earray(fout.root, 'X', atom=tables.Atom.from_dtype(np.array([0.5]).dtype), shape=(0, 1, n_features, n_time))
		X_scaled = fout.create_earray(fout.root, 'X_scaled', atom=tables.Atom.from_dtype(np.array([0.5]).dtype), shape=(0, 1, n_features, n_time))
		Z = fout.create_earray(fout.root, 'Z', atom=tables.Atom.from_dtype(np.array([1]).dtype), shape=(0, 1, n_features, n_time))
		Y = fout.create_earray(fout.root, 'Y', atom=tables.Atom.from_dtype(np.array([1]).dtype), shape=(0, n_outcomes, 1, 1))
		P = fout.create_earray(fout.root, 'P', atom=tables.Atom.from_dtype(np.array(['0123456789']).dtype), shape=(0,))
 
		X_vals = np.zeros((9, 1, n_features, n_time))
		X_scaled_vals = np.zeros((9, 1, n_features, n_time))
		Z_vals = np.zeros((9, 1, n_features, n_time))
		Y_vals = np.zeros((9, 1, 1, 1))
		Y_vals[1:5, 0, 0, 0] = 1

		X_vals[0,0,age_index,:] = 40.
		X_vals[1:,0,age_index,:] = 80.
		X_vals[0,0,gender_index,:] = 1.
		X_vals[:,0,0,0] = 25.
		X_vals[:,0,1,4] = 18. 
		X_vals[:,0,1,8] = 20.
		X_vals[:,0,1,10] = 22.

		m = np.mean([40] + [80]*8)
		s = float(np.std([40] + [80]*8))
		X_scaled_vals[0,0,age_index,:] = (40. - m)/s
		X_scaled_vals[1:,0,age_index,:] = (80. - m)/s
		X_scaled_vals[0,0,gender_index,:] = 1.
		X_scaled_vals[:,0,1,4] = (18. - 20.)/np.std([18,20,22])
		X_scaled_vals[:,0,1,10] = (22. - 20.)/np.std([18,20,22])

		Z_vals[:,0,age_index,:] = 1
		Z_vals[:,0,gender_index,:] = 1
		Z_vals[:,0,0,0] = 1
		Z_vals[:,0,1,4] = 1
		Z_vals[:,0,1,8] = 1
		Z_vals[:,0,1,10] = 1

		X.append(X_vals)
		X_scaled.append(X_scaled_vals)
		Z.append(Z_vals)
		Y.append(Y_vals)
		P.append(np.array(people[1:10]))
	
	loinc_db.close()
	loinc_vals_db.close()
	cpt_db.close()
	icd9_proc_db.close()
	icd9_db.close()
	ndc_db.close()

def features_assert_equals(a_fn, b_fn):

	with tables.open_file(a_fn, mode='r') as fin:
		a_X = fin.root.X[:]
		a_X_scaled = fin.root.X_scaled[:]
		a_Z = fin.root.Z[:]
		a_Y = fin.root.Y[:]
		a_P = fin.root.P[:]

	with tables.open_file(b_fn, mode='r') as fin:
		b_X = fin.root.X[:]
		b_X_scaled = fin.root.X_scaled[:]
		b_Z = fin.root.Z[:]
		b_Y = fin.root.Y[:]
		b_P = fin.root.P[:]

	assert (a_X == b_X).all()
	assert (a_X_scaled == b_X_scaled).all()
	assert (a_Z == b_Z).all()
	assert (a_Y == b_Y).all()
	assert (a_P == b_P).all()

def assert_equals(a, b, sort_by_col):
	assert len(a.columns) == len(b.columns)
	assert (np.sort(a.columns.values) == np.sort(b.columns.values)).all()
	assert len(a) == len(b)
	a = a.sort(sort_by_col)
	b = b.sort(sort_by_col)
	for col in a.columns:
		assert (a[col].values == b[col].values).all() 

def test():

	create_db()

	out_dir = tests_dir + 'kidney_disease/'

	if os.path.exists(tests_dir) == False:
		os.mkdir(out_dir)

	test_data_paths_fname = tests_dir + 'test_data_paths.yaml'
	test_stats_list_fname = tests_dir + 'test_stats.yaml'

	ckd.run(out_dir, test_data_paths_fname, test_stats_list_fname, use_just_common_labs=False, check_if_file_exists=False, verbose=False)

	test_soln_fnames = []
	test_soln_fnames.append(('kidney_disease_first_dialysis_cpt.txt', 'first_dialysis_cpt.txt'))
	test_soln_fnames.append(('kidney_disease_first_kidney_transplant_cpt.txt', 'first_kidney_transplant_cpt.txt'))
	test_soln_fnames.append(('kidney_disease_first_dialysis_icd9_proc.txt', 'first_dialysis_icd9_proc.txt'))
	test_soln_fnames.append(('kidney_disease_first_kidney_transplant_icd9_proc.txt', 'first_kidney_transplant_icd9_proc.txt'))
	test_soln_fnames.append(('kidney_disease_min_gfr.txt', 'min_gfr.txt'))
	test_soln_fnames.append(('kidney_disease_n_gap_stage45.txt', 'n_gap_stage45.txt'))
	test_soln_fnames.append(('kidney_disease_first_kidney_failure.txt', 'first_kidney_failure.txt'))
	test_soln_fnames.append(('kidney_disease_training_data.txt', 'training_data.txt'))
	for test_fname, soln_fname in test_soln_fnames:
		a = pd.read_csv(out_dir + test_fname, sep='\t', dtype=str)
		b = pd.read_csv(soln_dir + soln_fname, sep='\t', dtype=str)
		assert_equals(a, b, sort_by_col='person')

	features_assert_equals(out_dir + 'kidney_disease_features.h5', soln_dir + 'features.h5')
