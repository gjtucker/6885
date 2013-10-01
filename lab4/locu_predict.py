import utils
import hungarian
import numpy as np
import sys
import cPickle
from sklearn.externals import joblib

def weights_to_matching(p, index):
    # Convert the indices to numbers
    index_a = {}
    reverse_a = []
    index_b = {}
    reverse_b = []

    for (k_a, k_b) in index:
        if k_a not in index_a:
            index_a[k_a] = len(index_a)
            reverse_a.append(k_a)
        if k_b not in index_b:
            index_b[k_b] = len(index_b)
            reverse_b.append(k_b)

    weights = np.zeros((len(index_a), len(index_b)))
    for k in range(len(p)): 
        (k_a, k_b) = index[k]
        weights[index_a[k_a]][index_b[k_b]] = p[k][1]
        
    matching = hungarian.lap(-weights)[0]

    # matches are stored in a dictionary
    res = {}
    for i in range(len(matching)):
        locu = reverse_a[i]
        four = reverse_b[matching[i]]
        w = weights[i][matching[i]]
        res[(locu, four)] = w
    return res

# ----------------
# Main
# ----------------
# Load in the json files
model_file = sys.argv[1]
locu_file = sys.argv[2]
four_file = sys.argv[3]
match_out_file = sys.argv[4]

locu = utils.load_json(locu_file)  
four = utils.load_json(four_file) 

sys.stderr.write( "Featurizing dataset..." )
(X, index) = utils.featurize(locu, four, utils.sim)
sys.stderr.write( "done.\n" )

# Load in model
(model, thresh) = joblib.load(model_file)

## Test on testing
p = model.predict_proba(X)
res = weights_to_matching(p, index)
utils.write_matching(res, thresh, match_out_file)
