# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy

# <codecell>

def compute_equal_name( locu_record, foursquare_record ):
    return compute_equal_field( locu_record, foursquare_record, "name" )

def compute_equal_website( locu_record, foursquare_record ):
    return compute_equal_field( locu_record, foursquare_record, "website" )

def compute_equal_field( locu_record, foursquare_record, field_name ):
    if locu_record[field_name] == foursquare_record[field_name]:
        return 1.0
    else:
        return 0.0
def jaccard_strings( field1, field2 ):
    name1 = field1
    name2 = field2
    
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

def get_street_without_number( record ):
    street_address = record['street_address']
    if len( street_address  )== 0:
           return street_address
    else:
        try:
            float( street_address.split()[0] )
            return " ".join( street_address.split()[1:] )
        except ValueError:
            return street_address
        
# <codecell>

# <codecell>

def char_splitter(s, n):
    res = set()
    if len(s) < n:
        res.add(s)
        return res

    for i in range(len(s) - n + 1):
        res.add(s[i:(i+n)])
    return res


def jaccard_score(p1,p2,fields = ["name", "street_address" ], n = 4):
    name1 = " ".join([p1[x] for x in fields])
    name2 = " ".join([p2[x] for x in fields])
    set1 = set.union(*[char_splitter(x, n) for x in name1.lower().split()])
    set2 = set.union(*[char_splitter(x, n) for x in name2.lower().split()])
    c = set1.intersection(set2)
    return float(len(c)) / (len(set1) + len(set2) - len(c))

# <codecell>

def get_feature_vector( record1, record2 ):
    # return a numpy vector of floats
    feature_vector = []
    for fn in [ compute_equal_name, compute_equal_website, jaccard_score ]:
        x = fn( record1, record2 )
    
        feature_vector.append( numpy.float( x ) )
    
    return numpy.array( feature_vector )
    
    

    
    

    

# <codecell>

if __name__ == '__main__':
    data_path = '../'
    import os, json
    
    with open( os.path.join( data_path, 'foursquare_train_hard.json' ) ) as f:
        foursquare_records_train = json.load( f )
        
    with open( os.path.join( data_path, 'locu_train_hard.json' ) ) as f:
        locu_records_train = json.load( f )
    
    
    r1 = foursquare_records_train[0]
    r2 = locu_records_train[0]
    
    #print compute_equal_name( r1, r2 )
    
    features = get_feature_vector( r1, r2 )
    
    for fr in foursquare_records_train[:5]:
        for lr in locu_records_train[:5]:
            print get_feature_vector( fr, lr )
            
            #if numpy.sum( get_feature_vector( fr, lr ) ) == 2:
            #    print fr, lr
            
    
        

# <codecell>


