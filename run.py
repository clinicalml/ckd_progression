import pandas as pd
import numpy as np
import datetime as dt

import patient_stats as ps
ps = reload(ps)

import util
util = reload(util)

def run(out_dir, data_paths_fname, stats_list_fname, stats_key, check_if_file_exists, verbose=True): 

	# Load data

	db = util.Database(data_paths_fname)
	db.load_people()
	db.load_db(['loinc','loinc_vals','cpt','icd9_proc'])

	stats = util.read_yaml(stats_list_fname)[stats_key]

	# Calc patient stats

	data = ps.patient_stats(db, stats, stats_key, out_dir, stat_indices=None, verbose=verbose, check_if_file_exists=check_if_file_exists, save_files=True)
