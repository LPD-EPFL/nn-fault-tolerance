import numpy as np
import pickle
import pandas as pd

# Loading data
nProc = 3
results = []
for i in range(nProc):
    results += pickle.load(open('results_%d.pkl' % i, 'rb'))

# filling in data
for pt in results:
    pt['mean_exp'], pt['std_exp'], pt['mean_bound'], pt['std_bound'], _, _, bv2, pt['mean_v3_mean'] = pt['run']
    pt['mean_v2_mean'], pt['mean_v2_std'] = np.mean(bv2), np.std(bv2)
    del pt['run']

# res: key -> array of values
res = {k: [] for k in results[0].keys()}

# populating res
for pt in results:
    for key, value in pt.items():
        res[key] += [value]

# dataframe
pd.DataFrame(res).to_csv('results.csv')
