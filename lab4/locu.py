import utils
import hungarian
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

#    for (k, v) in pred_matching.iteritems():
#        if locu[i]["id"] in true_matching and \
#                (true_matching[locu[i]["id"]] == \
#                    four[j]["id"]):
#            truePos = truePos + 1
#        else:
#            falsePos = falsePos + 1
            #utils.print_obj(true_matching)
            #utils.print_obj(four[j])
            #print("\n")

    precision = truePos / float(truePos + falsePos)
    recall = truePos / float(len(true_matching))
    fmeas = (2.0 * precision * recall) / (precision + recall)

    print "TP = ",truePos,"FP = ",falsePos,"PREC = ",precision,"RECALL = ",recall,"F = ",fmeas

    return missed

def featurize(locu, four, sim):
    X = []
    index = []
    for (k_a, a) in locu.iteritems():
        for (k_b, b) in four.iteritems():
            X.append(sim(a, b))
            index.append((k_a, k_b))
    return (X, index)

def get_y(index, matches):
    y = [] 

    for (k_a, k_b) in index:
        if (k_a, k_b) in matches:
            y.append(True)
        else:
            y.append(False)
    return y

# ----------------
# Main
# ----------------
# Load in the json files
locu = utils.load_json('locu_train_hard.json')  
four = utils.load_json('foursquare_train_hard.json') 

locu_test = utils.load_json('locu_test_hard.json')  
four_test = utils.load_json('foursquare_test_hard.json') 

locu_easy = utils.load_json("locu_train.json")
four_easy = utils.load_json("foursquare_train.json")

# Read in matchs
matches_hard = utils.read_matches("matches_train_hard.csv")

# Remove crappy data from gold standard
del matches_hard[("5f3fd107090d0ddc658b", "51ce011a498ed8dfb15381bb")]
del matches_hard[("c170270283ef870d546b", "51eb7eed498e401ec51196b6")]
del matches_hard[("493f5e2798de851ec3b2", "51f119e7498e9716f71f4413")]
del matches_hard[("212dffb393f745df801a", "51e869ac498e7e485cabcdeb")]
del matches_hard[("e3f9d84c0c989f2e7928", "51e25e57498e535de72f03e7")]
del matches_hard[("66ef54d76ff989a91d52", "51c9e1dd498e33ecd8670892")]
del matches_hard[("edeba23f215dcc702220", "51a11cbc498e4083823909f1")]

matches_easy = utils.read_matches("matches_train.csv")

def sim(x, y):
    return [utils.jaccard_char_score(x, y, ["name"]), \
            utils.jaccard_char_score(x, y, ["street_address"]), \
            utils.jaccard_score(x, y, "street_address"), \
            utils.compute_equal_phones(x, y), \
            utils.distance(x, y), \
            1 if (x["phone"] is None) != (y["phone"] is None) else 0, \
            1 if (x["street_address"] is None) != (y["street_address"] is None) else 0, \
            1 if (x["name"] == y["name"]) else 0]

(X_easy, index_easy) = featurize(locu_easy, four_easy, sim)
y_easy = get_y(index_easy, matches_easy)

(X, index) = featurize(locu, four, sim)
y = get_y(index, matches_hard) 
#clf = LogisticRegression() 
clf = RandomForestClassifier(n_estimators = 64, n_jobs = 4)

X_tot = X + X_easy
y_tot = y + y_easy

clf = clf.fit(X_tot, y_tot)

## Test on training
p = clf.predict_proba(X)
res = weights_to_matching(p, index)

for thresh in np.linspace(0, .9, 20): 
    print(thresh)
    missed = score_matching(res, matches_hard, thresh)
"""
    # Print missed matches
    for (k_a, k_b) in missed:
        utils.print_obj(locu[k_a])
        utils.print_obj(four[k_b])
        print(res[(k_a, k_b)])
        print("\n")
"""
## Test on testing
thresh = 0.4
(X_test, index_test) = featurize(locu_test, four_test, sim)
p = clf.predict_proba(X_test)

res = weights_to_matching(p, index_test)
utils.write_matching(res, thresh)
