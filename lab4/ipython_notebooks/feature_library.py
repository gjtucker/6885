# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy

# <codecell>

def compute_equal_name( locu_record, foursquare_record ):
    if locu_record['name'] == foursquare_record['name']:
        return 1.0
    else:
        return 0.0

# <codecell>

def get_feature_vector( record1, record2 ):
    # return a numpy vector of floats
    feature_vector = []
    for fn in [ compute_equal_name ]:
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
    
    for fr in foursquare_records_train[:10]:
        for lr in locu_records_train[:10]:
            
        
    
        

# <codecell>


