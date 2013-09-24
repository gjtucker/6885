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
locu = load_json('locu_train.json')  
four = load_json('foursquare_train.json') 

locu_test = load_json('locu_test.json')  
four_test = load_json('foursquare_test.json') 

# Read in matchs
matches = {}
with file("matches_train.csv", 'r') as f:
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
    set1 = set.union(*[char_splitter(x, n) for x in name1.lower().split()])
    set2 = set.union(*[char_splitter(x, n) for x in name2.lower().split()])
    c = set1.intersection(set2)
    return float(len(c)) / (len(set1) + len(set2) - len(c))

def compute_weight_matrix(locu, four, score):
    res = np.zeros((len(locu), len(four)))

    for i in range(len(locu)):
        for j in range(len(four)):
            res[i][j] = score(locu[i], four[j])

    return res

def predict(locu, four, score):
    weights = compute_weight_matrix(locu, four, score)
    matching = hungarian.lap(-weights)[0]

    return matching

def write_matching(locu, four, matching, file_name = "matches_test.csv", debug = False):
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

    for i in range(len(pred_matching)):
        if (true_matching[locu[i]["id"]] == four[pred_matching[i]]["id"]):
            truePos = truePos + 1
        else:
            falsePos = falsePos + 1

    precision = truePos / float(truePos + falsePos)
    recall = truePos / float(len(locu))
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
"""

matching = predict(locu, four, \
                lambda x,y: jaccard_score(x, y, ["name", "street_address"]))
score_matching(locu, four, matching, matches)

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
