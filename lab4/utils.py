import math
import editdist
import json

def print_obj(json_obj):
    print json.dumps(json_obj, sort_keys = True, indent = 4, separators = (',', ': '))

def load_json(file_name):
    with file(file_name, 'r') as f:
        res = json.load(f)
    return res

def write_matching(locu, four, matching, file_name = "matches_test_hard.csv", debug = False):
    with open(file_name, 'w') as out:
        out.write("locu_id,foursquare_id\n")
        for (i, j) in matching:
            out.write("%s,%s\n" % (locu[i]["id"], four[j]["id"])) 
            
            if debug:
                print_obj(locu[i])
                print_obj(four[j])
                print("\n")


#### Features
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

def compute_equal_phones(x, y):
    phone_x = x["phone"]
    phone_y = y["phone"]

    if phone_x is None and phone_y is None:
        return 1
    if phone_x is None or phone_y is None:
        return 0
    phone_y = phone_y[1:4] + phone_y[6:9] + phone_y[10:]
    return 1 if phone_x == phone_y else 0 

def distance(p1, p2):
    lat1 = p1["latitude"]
    lon1 = p1["longitude"]
    lat2 = p2["latitude"]
    lon2 = p2["longitude"]

    if lat1 is None:
        return 10

    x = (lon2 - lon1) * math.cos((lat1 + lat2)/2)
    y = lat2 - lat1
    return math.sqrt(x*x + y*y)

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


