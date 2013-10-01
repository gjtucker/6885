import utils
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import sys
from sklearn.externals import joblib
import cPickle
from tfidf import Tfidf

def run_classifier_learning(X, y, models, file_name, n_jobs = 5, \
        debug = False):
    # Model selection with CV
    best_score = 0
    best_model = None
    for (name, model) in models.iteritems():
        sys.stderr.write("CV on model %s\n" % name)
        score = np.mean(cross_validation.cross_val_score(model, X, y, \
                cv = 10, n_jobs = n_jobs))
        if score > best_score:
            best_score = score
            best_model = name 
   
    sys.stderr.write("Chose model %s with f-score = %g\n" % \
            (best_model, best_score))
    best_model = models[name]
    
    # Fit best model to the whole training set
    best_model.fit(X, y)

    # Pickle to file
    joblib.dump(best_model, file_name) 
    
    # If debugging print out extra useful information
    if debug:
        # Print out feature importances
        sys.stderr.write("Feature importances:\n")
        for i in range(len(best_model.feature_importances_)):
            sys.stderr.write("%dth feature importance: %g" % (i, \
                    best_model.feature_importances_[i]))

        # Print out the top k ranked negatives and the bottom ranked 
        # positives, TODO
    
    return (best_model, best_score)

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

locu_easy_test = utils.load_json("locu_test.json")
four_easy_test = utils.load_json("foursquare_test.json")

# Read in matches
matches_hard = utils.read_matches("matches_train_hard.csv")

#get word statistics for IDF-type features                                                                                              
sys.stderr.write( "Getting tf-idf statistics..." )
tfidf_obj = Tfidf( locu, four, "name" )
sys.stderr.write( "done.\n" )

# Remove crappy data from gold standard
del matches_hard[("5f3fd107090d0ddc658b", "51ce011a498ed8dfb15381bb")]
del matches_hard[("c170270283ef870d546b", "51eb7eed498e401ec51196b6")]
del matches_hard[("493f5e2798de851ec3b2", "51f119e7498e9716f71f4413")]
del matches_hard[("212dffb393f745df801a", "51e869ac498e7e485cabcdeb")]
del matches_hard[("e3f9d84c0c989f2e7928", "51e25e57498e535de72f03e7")]
del matches_hard[("66ef54d76ff989a91d52", "51c9e1dd498e33ecd8670892")]
del matches_hard[("edeba23f215dcc702220", "51a11cbc498e4083823909f1")]

# Extra matches
matches_hard_test = utils.read_matches("matches_test_hard.csv")
matches_easy = utils.read_matches("matches_train.csv")
matches_easy_test = utils.read_matches("matches_test.csv")

# Compiling data sets
try:
    f = open("working/locu_classifier.cache", 'rb')
    (X_tot, y_tot) = cPickle.load(f)
    sys.stderr.write("Loading data from cache.")
except IOError:
    sys.stderr.write( "Featurizing easy dataset..." )
    (X_easy, index_easy) = utils.featurize(locu_easy, four_easy, utils.sim)
    y_easy = utils.get_y(index_easy, matches_easy)
    sys.stderr.write( "done.\n" )

    sys.stderr.write( "Featurizing easy test dataset..." )
    (X_easy_test, index_easy_test) = utils.featurize(locu_easy_test, four_easy_test, utils.sim)
    y_easy_test = utils.get_y(index_easy_test, matches_easy_test) 
    sys.stderr.write( "done.\n" )

    sys.stderr.write( "Featurizing hard dataset..." )
    (X, index) = utils.featurize(locu, four, utils.sim)
    y = utils.get_y(index, matches_hard) 
    sys.stderr.write( "done.\n" )

    sys.stderr.write( "Featurizing hard test dataset..." )
    (X_hard_test, index_hard_test) = utils.featurize(locu_test, four_test, utils.sim)
    y_hard_test = utils.get_y(index_hard_test, matches_hard_test) 

    X_tot = X + X_easy + X_easy_test + X_hard_test
    y_tot = y + y_easy + y_easy_test + y_hard_test
    
    with open("working/locu_classifier.cache", "wb") as out:
        cPickle.dump((X_tot, y_tot), out)

# Learning

classifier_model_file = sys.argv[1]

sys.stderr.write( "Fitting classifier..." )
models = { "RF trees = 64, auto" : RandomForestClassifier(n_estimators = 64, \
                n_jobs = 10), \
           "RF trees = 64, sqrt" : RandomForestClassifier(n_estimators = 64, \
                n_jobs = 10, max_features = "sqrt"), \
           "RF trees = 64, log2" : RandomForestClassifier(n_estimators = 64, \
                n_jobs = 10, max_features = "log2"), \
        }
(model, score) = run_classifier_learning(X_tot, np.array(y_tot), \
        models, classifier_model_file, n_jobs = 1)

sys.stderr.write( "done.\n" )
