from sklearn.ensemble import RandomForestClassifier
import math
import json
import editdist
import hungarian
import numpy as np

"""
Required packages:
    numpy
    hungarian
    json

"""

def print_obj(json_obj):
    print json.dumps(json_obj, sort_keys = True, indent = 4, separators = (',', ': '))

def load_json(file_name):
    with file(file_name, 'r') as f:
        res = json.load(f)
    return res

# Load in the json files
# Load in the matchs
locu = load_json('locu_train_hard.json')  
four = load_json('foursquare_train_hard.json') 

locu_test = load_json('locu_test_hard.json')  
four_test = load_json('foursquare_test_hard.json') 

# Read in matchs
matches = {}
with file("matches_train_hard.csv", 'r') as f:
    next(f)
    for line in f:
        (locu_id, four_id) = line.strip().split(",")
        matches[locu_id] = four_id

def string_match_score(p1,p2,field):
    s1 = p1[field]
    s2 = p2[field]
    return -editdist.distance(s1.lower(),s2.lower())/float(len(s1))

def char_splitter(s, n):
    res = set()
    if len(s) < n:
        res.add(s)
        return res

    for i in range(len(s) - n + 1):
        res.add(s[i:(i+n)])
    return res

def distance(p1, p2):
    lat1 = p1["latitude"]
    lon1 = p1["longitude"]
    lat2 = p2["latitude"]
    lon2 = p2["longitude"]
    
    x = (lon2 - lon1) * math.cos((lat1 + lat2)/2)
    y = lat2 - lat1
    return -math.sqrt(x*x + y*y)

def jaccard_score(p1,p2,fields, n = 4):
    name1 = " ".join([p1[x] for x in fields])
    name2 = " ".join([p2[x] for x in fields])
    
    if name1 == "":
        set1 = set()
    else:
        set1 = set.union(*[char_splitter(x, n) for x in name1.lower().split()])
    if name2 == "":
        set2 = set()
    else:
        set2 = set.union(*[char_splitter(x, n) for x in name2.lower().split()])

    c = set1.intersection(set2)
    denom = (len(set1) + len(set2) - len(c))

    return 0 if denom == 0 else float(len(c)) / denom

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

def write_matching(locu, four, matching, file_name = "matches_test_hard.csv", debug = False):
    with open(file_name, 'w') as out:
        out.write("locu_id,foursquare_id\n")
        for i in range(len(matching)):
            out.write("%s,%s\n" % (locu[i]["id"], four[matching[i]]["id"])) 
            
            if debug:
                print_obj(locu[i])
                print_obj(four[matching[i]])
                print("\n")

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
            print_obj(locu[i])
            print_obj(four[matching[i]])
            print("\n")


    precision = truePos / float(truePos + falsePos)
    recall = truePos / float(len(true_matching))
    fmeas = (2.0 * precision * recall) / (precision + recall)

    print "TP = ",truePos,"FP = ",falsePos,"PREC = ",precision,"RECALL = ",recall,"F = ",fmeas

# ----------------
# Main
# ----------------

"""
print "Matching"
matching = predict(locu_test, four_test, \
                lambda x,y: jaccard_score(x, y, ["name", "street_address"]))

print "Writing matching"
write_matching(locu_test, four_test, matching)

res = predict(locu, four, \
          lambda x,y: jaccard_score(x, y, ["name", "street_address"]))

for thresh in np.linspace(0, 0.9, 20):
    truncated_matches = [x[1] for x in res if x[0] > thresh]
    score_matching(locu, four, truncated_matches, matches)


Generate training data
Have to use CV? For now no, hopefully no overfitting.
"""

def sim(x, y):
    return [jaccard_score(x, y, ["name"]), \
            jaccard_score(x, y, ["street_address"])]

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

(X, y, index) = featurize(locu, four, sim)
clf = RandomForestClassifier(n_estimators = 10)
clf = clf.fit(X, y)
p = clf.predict_proba(X)

weights = np.zeros((len(locu), len(four)))
for k in range(len(p)): 
    (i, j) = index[k]
    weights[i][j] = p[k][1]
    
matching = hungarian.lap(-weights)[0]
res = [(weights[i][matching[i]], (i, matching[i])) \
        for i in range(len(matching))]

for thresh in np.linspace(0, 0.9, 10):
    truncated_matches = [x[1] for x in res if x[0] > thresh]
    score_matching(locu, four, truncated_matches, matches)




"""
for loop in range(0,10,1):
    falsePos = 0
    truePos = 0
    falseNeg = 0
    trueNeg = 0
    thresh = float(loop)/10.0

    for r1 in locu:
        bestMatch = 0 
        bestVal = []
        j = 0
        for r2 in four:
            s = jaccard_score(r1,r2,"name")
            #s = distance(r1,r2)
            if (s > bestMatch):
                bestMatch = s
                bestVal = r2
        if (bestMatch > thresh):
            #        print "Best match: ",r1["name"],bestVal["name"],"score=",bestMatch
            if (matches[r1["id"]] == bestVal["id"]):
                truePos = truePos + 1
            else:
                print bestMatch
                print_obj(r1)
                print_obj(bestVal) 
                print("\n")
                falsePos = falsePos + 1

    precision = truePos / float(truePos + falsePos)
    recall = truePos / float(len(locu))
    fmeas = (2.0 * precision * recall) / (precision + recall)

    print "THRESH = ",thresh,"TP = ",truePos,"FP = ",falsePos,"PREC = ",precision,"RECALL = ",recall,"F = ",fmeas
    """
