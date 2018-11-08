import numpy as np
import pickle
import pandas as pd

# Loading data
nProc = 3
results = []
for i in range(nProc):
    results += pickle.load(open('results_%d.pkl' % i, 'rb'))

# filling in l1, l2
for pt in results:
    l1 = 0
    l2 = 0
    for W in pt['W']:
        l1 += np.linalg.norm(W, ord = 1)
        l2 += np.linalg.norm(W, ord = 2)
    for B in pt['B']:
        l1 += np.linalg.norm(B, ord = 1)
        l2 += np.linalg.norm(B, ord = 2)
    pt['l1'] = l1
    pt['l2'] = l2

# removing W, B
for pt in results:
    del pt['W'], pt['B']

# filling in data
for pt in results:
    pt['mean_exp'], pt['std_exp'], _, _, _, _, _ = pt['run']
    pt['mean_bound'], pt['std_bound'] = pt['mean_std']
    pt['mean_v2_mean'] = np.mean(pt['mean_v2'])
    pt['mean_v2_std'] = np.std(pt['mean_v2'])
    del pt['run'], pt['mean_std'], pt['mean_v2']

# res: key -> array of values
res = {k: [] for k in results[0].keys()}

# populating res
for pt in results:
    for key, value in pt.items():
        res[key] += [value]

# dataframe
pd.DataFrame(res).to_csv('results.csv')
