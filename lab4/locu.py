import utils
import hungarian
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def compute_weight_matrix(locu, four, score):
    res = np.zeros((len(locu), len(four)))

    for i in range(len(locu)):
        for j in range(len(four)):
            res[i][j] = score(locu[i], four[j])

    return res

def predict(locu, four, score):
    weights = compute_weight_matrix(locu, four, score)
    matching = hungarian.lap(-weights)[0]
    res = [(weights[i][matching[i]], (i, matching[i])) \
            for i in range(len(matching))]

    return res 

def weights_to_matching(p, index, size):
    weights = np.zeros((size, size))
    for k in range(len(p)): 
        (i, j) = index[k]
        weights[i][j] = p[k][1]
        
    matching = hungarian.lap(-weights)[0]
    res = [(weights[i][matching[i]], (i, matching[i])) \
            for i in range(len(matching))]
    return res

def score_matching(locu, four, pred_matching, true_matching):
    falsePos = 0
    truePos = 0

    for (i,j) in pred_matching:
        if locu[i]["id"] in true_matching and \
                (true_matching[locu[i]["id"]] == \
                    four[j]["id"]):
            truePos = truePos + 1
        else:
            falsePos = falsePos + 1
            #utils.print_obj(true_matching)
            #utils.print_obj(four[j])
            #print("\n")

    precision = truePos / float(truePos + falsePos)
    recall = truePos / float(len(true_matching))
    fmeas = (2.0 * precision * recall) / (precision + recall)

    print "TP = ",truePos,"FP = ",falsePos,"PREC = ",precision,"RECALL = ",recall,"F = ",fmeas

def featurize(locu, four, sim):
    X = []
    y = [] 
    index = []
    for i in range(len(locu)):
        for j in range(len(four)):
            X.append(sim(locu[i], four[j]))
            if locu[i]["id"] in matches:
                y.append(matches[locu[i]["id"]] == four[j]["id"])
            else:
                y.append(False)
            index.append((i, j))
    return (X, y, index)


# ----------------
# Main
# ----------------
# Load in the json files
locu = utils.load_json('locu_train_hard.json')  
four = utils.load_json('foursquare_train_hard.json') 

locu_test = utils.load_json('locu_test_hard.json')  
four_test = utils.load_json('foursquare_test_hard.json') 

# Read in matchs
matches = {}
with file("matches_train_hard.csv", 'r') as f:
    next(f)
    for line in f:
        (locu_id, four_id) = line.strip().split(",")
        matches[locu_id] = four_id

def sim(x, y):
    return [utils.jaccard_score(x, y, ["name"]), \
            utils.jaccard_score(x, y, ["street_address"]), \
            utils.compute_equal_phones(x, y), \
            utils.distance(x, y)]

(X, y, index) = featurize(locu, four, sim)
clf = RandomForestClassifier(n_estimators = 64)
clf = clf.fit(X, y)


## Test on training
"""
p = clf.predict_proba(X)
res = weights_to_matching(p, index, len(locu))

for thresh in np.linspace(0, 0.9, 10): 
    truncated_matches = [x[1] for x in res if x[0] > thresh]
    print(thresh)
    score_matching(locu, four, truncated_matches, matches)
"""

## Test on testing
thresh = 0.2
(X_test, _, index_test) = featurize(locu_test, four_test, sim)
p = clf.predict_proba(X_test)

res = weights_to_matching(p, index_test, len(locu_test))
truncated_matches = [x[1] for x in res if x[0] > thresh]
utils.write_matching(locu_test, four_test, truncated_matches)

