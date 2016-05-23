import pylab as pl
import numpy as np
import pandas as pd
import shelve
import pdb
import pickle
import datetime as dt
import time
import yaml
import copy
import scipy.sparse

def calc_gfr(val, age, is_female):

	if val == 0:
		return 0

	# more likely that the lab value is expressed in mg/L rather than mg/dL and the patient has a GFR of ~180 than ~0.8
	if val >= 60:
		val /= 100.

	gfr_val = 175 * (val ** (-1.154)) * (age**(-0.203)) 
	gfr_val *= ((1. + 1.212)/2.) # average btwn races since we don't have race
	if is_female == True:
		gfr_val *= 0.742

	if gfr_val == 0:
		return 0

	if gfr_val < 1:
		return 0

	if gfr_val > 300:
		return 0

	return gfr_val

def create_ndc_descs(out_fname, data_paths):

	ndcs = read_list_files(data_paths['ndc_list_fname'])
	ndc_refs = pd.read_csv(data_paths['ndc_ref_fname'])

	ndc_refs['NDC_STR'] = ndc_refs['NDC_CD'].apply(str)
	ndc_refs['NDC_LEN'] = ndc_refs['NDC_STR'].apply(len)
	ndc_refs['NDC'] = ''
	ndc_refs.loc[ndc_refs['NDC_LEN'] == 7, 'NDC'] = '0000' + ndc_refs['NDC_STR'] 
	ndc_refs.loc[ndc_refs['NDC_LEN'] == 8, 'NDC'] = '000' + ndc_refs['NDC_STR'] 
	ndc_refs.loc[ndc_refs['NDC_LEN'] == 9, 'NDC'] = '00' + ndc_refs['NDC_STR'] 
	ndc_refs.loc[ndc_refs['NDC_LEN'] == 10, 'NDC'] = '0' + ndc_refs['NDC_STR'] 
	ndc_refs.loc[ndc_refs['NDC_LEN'] == 11, 'NDC'] = ndc_refs['NDC_STR']

	ndc_refs_small = ndc_refs[['NDC','NDC_CD','BRAND_NM','GENRC_LONG_NM','AHFS_TC_1_DSC']].drop_duplicates()	
	ndc_descs = pd.DataFrame({'NDC': ndcs})
	ndc_descs = pd.merge(ndc_descs, ndc_refs_small, on='NDC')
	ndc_descs.to_csv(out_fname, index=False, sep='\t')

	return ndc_descs

class Database:

	def __init__(self, data_paths_fname):
		self.data_paths_fname = data_paths_fname
		self.data_paths = read_yaml(data_paths_fname)
		self.codes = {}
		self.code_to_index = {}
		self.code_db = {}
		self.db = {}
		self.descs = {}
		self.person_to_index = {}
	
		self.demographics = {}

	def load_people(self):
		self.people = read_list_files(self.data_paths['people_list_fname'])
		#self.person_to_index = dict((person, index) for index, person in enumerate(self.people))

	def load_descs(self, desc):

		if desc == 'icd9':

			self.descs['icd9'] = {}

			with open(self.data_paths['icd9_descs_fname'], 'r') as fin:
				for line in fin:
					vals = line.strip().split('#')
					icd9 = vals[0]
					desc = vals[1].replace(icd9 + ' ', '')
					self.descs['icd9'][icd9] = desc

		elif desc == 'loinc':

			self.descs['loinc'] = {}
		
			with open(self.data_paths['loinc_descs_fname'], 'r') as fin:
				for line in fin:
					vals = line.strip().split('#')
					loinc = vals[0]
					desc = vals[1]
					self.descs['loinc'][loinc] = desc

		elif desc == 'cpt':
	
			self.descs['cpt'] = {}

			with open(self.data_paths['cpt_descs_fname'], 'r') as fin:
				for line in fin:
					vals = line.strip().split(',')
					cpt = vals[0]
					desc = vals[1].replace('"', '')
					self.descs['cpt'][cpt] = desc

		elif desc == 'ndc':

			ndc_descs = pd.read_csv(self.data_paths['ndc_descs_fname'], sep='\t', dtype={'NDC': 'S20', 'NDC_CD': 'S20', 'BRAND_NM': 'S20', 'GENRC_LONG_NM': 'S20', 'AHFS_TC_1_DSC': 'S500'})
			self.descs['ndc'] = dict((ndc_descs['NDC'].iloc[i], ndc_descs['BRAND_NM'].iloc[i] + '##' + ndc_descs['GENRC_LONG_NM'].iloc[i] + '##' + ndc_descs['AHFS_TC_1_DSC'].iloc[i]) for i in range(len(ndc_descs)))

	def load_db(self, db_keys, people=False, load_codes=False, load_descs=False):

		if people == True:
			self.load_people()

		if type(db_keys) == type(''):
			db_keys = [db_keys]

		for db_key in db_keys:
			assert (db_key in set(['icd9','loinc','loinc_vals','ndc','cpt','icd9_proc'])) == True

			self.codes[db_key] = read_list_files(self.data_paths[db_key + '_list_fname'])
			self.code_to_index[db_key] = dict((code, index) for index, code in enumerate(self.codes[db_key]))
			self.db[db_key] = shelve.open(self.data_paths[db_key + '_db_fname'])

			if db_key != 'loinc_vals' and load_codes == True:
				self.code_db[db_key] = shelve.open(self.data_paths[db_key + '_code_db_fname'])

			if (db_key in set(['icd9','loinc','cpt', 'ndc'])) == True and load_descs == True:
				self.load_descs(db_key)

def read_list_files(fname):
	with open(fname, 'r') as fin:
		data = fin.read().strip().split('\n')
	return data


def read_yaml(fname, bunch=False):
	with open(fname, 'r') as fin:
		data = yaml.load(fin)
		if bunch == True:
			data = Bunch(data)
	return data

class Bunch(object):
	def __init__(self, adict):
		self.__dict__.update(adict)
