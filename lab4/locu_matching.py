import utils
import hungarian
import numpy as np
import sys
from tfidf import Tfidf
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

def score_matching(pred_matching, true_matching, thresh):
    falsePos = 0
    truePos = 0
    missed = []

    for k in true_matching.iterkeys():
        if k in pred_matching and pred_matching[k] > thresh:
            truePos += 1
        else:
            missed.append(k)

    total_predictions = 0
    for v in pred_matching.itervalues():
        if v > thresh:
            total_predictions += 1

    falsePos = total_predictions - truePos

    precision = truePos / float(truePos + falsePos)
    recall = truePos / float(len(true_matching))
    fmeas = (2.0 * precision * recall) / (precision + recall)

    print "TP = ",truePos,"FP = ",falsePos,"PREC = ",precision,"RECALL = ",recall,"F = ",fmeas

    return fmeas 

# ----------------
# Main
# ----------------
# Load in the json files
locu = utils.load_json('locu_train_hard.json')  
four = utils.load_json('foursquare_train_hard.json') 

locu_test = utils.load_json('locu_test_hard.json')  
four_test = utils.load_json('foursquare_test_hard.json') 

#get word statistics for IDF-type features                                                                                              
sys.stderr.write( "Getting tf-idf statistics..." )
tfidf_obj = Tfidf( locu, four, "name" )
sys.stderr.write( "done.\n" )

# Read in matches
matches_hard = utils.read_matches("matches_train_hard.csv")

# Remove crappy data from gold standard
del matches_hard[("5f3fd107090d0ddc658b", "51ce011a498ed8dfb15381bb")]
del matches_hard[("c170270283ef870d546b", "51eb7eed498e401ec51196b6")]
del matches_hard[("493f5e2798de851ec3b2", "51f119e7498e9716f71f4413")]
del matches_hard[("212dffb393f745df801a", "51e869ac498e7e485cabcdeb")]
del matches_hard[("e3f9d84c0c989f2e7928", "51e25e57498e535de72f03e7")]
del matches_hard[("66ef54d76ff989a91d52", "51c9e1dd498e33ecd8670892")]
del matches_hard[("edeba23f215dcc702220", "51a11cbc498e4083823909f1")]

matches_hard_test = utils.read_matches("matches_test_hard.csv")

sys.stderr.write( "Featurizing hard dataset..." )
(X, index) = utils.featurize(locu, four, utils.sim)
y = utils.get_y(index, matches_hard) 
sys.stderr.write( "done.\n" )

sys.stderr.write( "Featurizing hard test dataset..." )
(X_hard_test, index_hard_test) = utils.featurize(locu_test, four_test, utils.sim)
y_hard_test = utils.get_y(index_hard_test, matches_hard_test) 

X_tot = X + X_hard_test
y_tot = y + y_hard_test

# Load in model
classifier_model_file = sys.argv[1]
matcher_model_file = sys.argv[2]

model = joblib.load(classifier_model_file)

# Run the trained model to select best threshold
sys.stderr.write( "Predicting..." )
p = model.predict_proba(X)
sys.stderr.write( "done.\n" )

# Get matching
res = weights_to_matching(p, index)

# Loop through thresholds
best_fmeas = 0 
best_thresh = [] 
for thresh in np.linspace(0, .9, 100): 
    print(thresh)
    fmeas = score_matching(res, matches_hard, thresh)

    if fmeas >= best_fmeas:
        if fmeas > best_fmeas:
            best_fmeas = fmeas
            best_thresh = [thresh]
        else:
            best_thresh.append(thresh)

# Take the mean threshold
best_thresh = np.mean(best_thresh)
sys.stderr.write("Best thresh = %g giving f measure = %g" % \
        (best_thresh, best_fmeas))

# Pickle the model and threshold
joblib.dump((model, best_thresh), matcher_model_file)


