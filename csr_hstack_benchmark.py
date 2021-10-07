import numpy as np
import time
import itertools
from scipy import sparse
from topn import awesome_hstack
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')


# init parameters
NROWS = int(2e6)
NCOLS_DENSE = 200
NCOLS_SPARSE = int(1e5) # approximate 
SPARSE_DENSITY = 2e-4

# Dense data creation
data_dense = np.random.uniform(size=[NROWS, NCOLS_DENSE]).astype(np.float64)

# Sparse data creation. 
words_in_sentence = int(NCOLS_SPARSE*SPARSE_DENSITY)
corpus = np.random.choice(size=[NROWS, words_in_sentence], a=NCOLS_SPARSE)
corpus = list((map(lambda x: ' '.join(x.astype(str)), corpus)))
print ('Example of 3 sentences for sparse matrix creation:')
for sentence in corpus[:3]:
    print (sentence)
print()

vectorizer = CountVectorizer().fit(corpus) 
data_sparse = vectorizer.transform(corpus).astype(np.float64)

print ('Dense  data shape:', data_dense.shape)
print ('Sparse data shape:', data_sparse.shape, '\n')


# check whether function returns the same result as default hstack
assert (sparse.hstack([data_sparse[:200], data_sparse[:200]], format='csr') 
        == awesome_hstack([data_sparse[:200], data_sparse[:200]], use_threads=True, n_jobs=7)).toarray().all()
assert (sparse.hstack([data_sparse[:200], data_sparse[:200]], format='csr') 
        == awesome_hstack([data_sparse[:200], data_sparse[:200]])).toarray().all()
        
number_of_trials = 4 #  Number of trials for each matrix size. Is used to calculate 95% confedence intervals
nrows_list = np.array([1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 7.5e5, 1e6, 1.25e6, 1.5e6, 1.75e6, 2e6]).astype(int)

default_time_means = []
default_time_stds  = []

awesome_time_means = []
awesome_time_stds  = []

for nrows in nrows_list:
    print ('Started on {} row'.format(nrows)) 
    
    default_trials = []
    awesome_trials = []
    
    for trial_num in range(number_of_trials):
        
        # Measuring scipy sparse hstack performance
        t_start = time.time()
        _ = sparse.hstack([data_sparse[:nrows], data_sparse[:nrows]], format='csr')
        default_trials.append(time.time() - t_start)
        
        # Measuring awesome_hstack performance
        t_start = time.time()
        _ = awesome_hstack([data_sparse[:nrows], data_sparse[:nrows]])
        awesome_trials.append(time.time() - t_start)
        
    default_time_means.append(np.mean(default_trials))
    awesome_time_means.append(np.mean(awesome_trials))
    
    default_time_stds.append( 2*np.std(default_trials, ddof=1)/np.sqrt(number_of_trials) )
    awesome_time_stds.append( 2*np.std(awesome_trials, ddof=1)/np.sqrt(number_of_trials) )
    
print(f'{default_time_means=}', flush=True)
print(f'{awesome_time_means=}', flush=True)
