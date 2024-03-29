import math
import json
from tfidf import Tfidf

def print_obj(json_obj):
    print json.dumps(json_obj, sort_keys = True, indent = 4, separators = (',', ': '))

def load_json(file_name):
    with file(file_name, 'r') as f:
        json_objects = json.load(f)

    res = {}
    for o in json_objects:
        res[o["id"]] = o

    return res

def read_matches(file_name):
    matches = {}
    with file(file_name, 'r') as f:
        next(f)
        for line in f:
            (locu_id, four_id) = line.strip().split(",")
            matches[(locu_id, four_id)] = 1 
    return matches

def write_matching(matching, thresh, file_name = "matches_test_hard.csv", debug = False):
    good = []
    with open(file_name, 'w') as out:
        out.write("locu_id,foursquare_id\n")
        for (k, v) in matching.iteritems():
            if v > thresh: 
                out.write("%s,%s\n" % k) 
                good.append((v, k))

    good.sort()
    print(good[0:20])

#### Features
def char_splitter(s, n):
    res = set()
    if len(s) < n:
        res.add(s)
        return res

    for i in range(len(s) - n + 1):
        res.add(s[i:(i+n)])
    return res

def compute_equal_phones(x, y):
    phone_x = x["phone"]
    phone_y = y["phone"]

    #if phone_x is None and phone_y is None:
    #    return 1
    if phone_x is None or phone_y is None:
        return 0
    phone_y = phone_y[1:4] + phone_y[6:9] + phone_y[10:]
    return 1 if phone_x == phone_y else 0 


def jaccard_words(p1, p2):
    # lat1 = p1["latitude"]
    # lon1 = p1["longitude"]
    # lat2 = p2["latitude"]
    # lon2 = p2["longitude"]

    # if lat1 is None:
    #     return 10

    # x = (lon2 - lon1) * math.cos((lat1 + lat2)/2)
    # y = lat2 - lat1
    # return math.sqrt(x*x + y*y)
    pass

def distance(p1, p2):
    lat1 = p1["latitude"]
    lon1 = p1["longitude"]
    lat2 = p2["latitude"]
    lon2 = p2["longitude"]

    if lat1 is None or lat2 is None:
        return 10

    x = (lon2 - lon1) * math.cos((lat1 + lat2)/2)
    y = lat2 - lat1
    return math.sqrt(x*x + y*y)

def jaccard_char_score(p1,p2,fields, n = 4):
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

def jaccard_score(p1,p2,field):
    name1 = p1[field] 
    name2 = p2[field]
    
    if name1 == "":
        set1 = set()
    else:
        set1 = set(name1.lower().split())
    if name2 == "":
        set2 = set()
    else:
        set2 = set(name2.lower().split())

    c = set1.intersection(set2)
    denom = (len(set1) + len(set2) - len(c))

    return 0 if denom == 0 else float(len(c)) / denom


def faster_jaccard_score_tfidf( p1, p2, field, tfidf ):
    name1 = p1[field] 
    name2 = p2[field]
    
    if name1 == "":
        set1 = set()
    else:
        set1 = set(name1.lower().split())
    if name2 == "":
        set2 = set()
    else:
        set2 = set(name2.lower().split())

    i = list(set1.intersection(set2))
    u = list(set1.union(set2))
    #compute idf score (decided to ignore tf)
    iscore = sum([tfidf.get_score(word) for word in i])
    uscore = sum([tfidf.get_score(word) for word in u])

    #import pdb
    #pdb.set_trace()

    return 0 if uscore == 0 else float(iscore) / uscore

def jaccard_score_tfidf(locu, four, p1,p2,field):
    #make a tfidf object so that we don't need to re-compute the list of all names every time
    tfidf = Tfidf(locu, four,field)

    name1 = p1[field] 
    name2 = p2[field]
    
    if name1 == "":
        set1 = set()
    else:
        set1 = set(name1.lower().split())
    if name2 == "":
        set2 = set()
    else:
        set2 = set(name2.lower().split())

    i = list(set1.intersection(set2))
    u = list(set1.union(set2))

    #compute idf score (decided to ignore tf)
    iscore = sum([tfidf.get_score(word) for word in i])
    uscore = sum([tfidf.get_score(word) for word in u])

    return 0 if uscore == 0 else float(iscore) / uscore

def is_none(x):
    return 1 if x is None else 0

def is_equal(x, y):
    if x is None or y is None:
        return 0
    return x == y

def sim(x, y):
    return [jaccard_char_score(x, y, ["name"]), \
            jaccard_char_score(x, y, ["street_address"]), \
            #utils.jaccard_score_tfidf(locu, four, x, y, "name"), \
            jaccard_score(x, y, "street_address"), \
            compute_equal_phones(x, y), \
            distance(x, y), \
            1 if (x["phone"] is None) != (y["phone"] is None) else 0, \
            1 if (x["street_address"] is None) != (y["street_address"] is None) else 0, \
            1 if (x["name"] == y["name"]) else 0]

def sim2(x, y):
    return [jaccard_char_score(x, y, ["name"]), \
            jaccard_char_score(x, y, ["street_address"]), \
            is_none(x["street_address"]), \
            is_none(y["street_address"]), \
            is_none(x["phone"]), \
            is_none(y["phone"]), \
            is_equal(x["postal_code"], y["postal_code"]), \
            #utils.jaccard_score_tfidf(locu, four, x, y, "name"), \
            jaccard_score(x, y, "street_address"), \
            compute_equal_phones(x, y), \
            distance(x, y), \
            1 if (x["name"] == y["name"]) else 0]

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

