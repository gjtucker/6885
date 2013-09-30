# this does tfidf
import math

class Tfidf:
    def __init__(self, locu, four, field):
        self.field = field
        self.listofdocs = self.get_listofdocs(locu, four, field)
        self.numdocs = len( self.listofdocs )
        trainingwords = set( [ i for sublist in self.listofdocs for i in sublist ] )
        self.docfreq_dict = {}
        for w in list( trainingwords ):
            self.docfreq_dict[w] = self.num_docs_containing( w, self.listofdocs )

        
    def get_cleanwords(self, str_input):
        return str_input.lower().split(" ")

    def get_listofdocs(self, locu, four, field):
        docs = []
        docs += [self.get_cleanwords(a[field]) for (k_a, a) in locu.iteritems()]
        docs += [self.get_cleanwords(b[field]) for (k_b, b) in four.iteritems()]


        return docs

    ## major public method
    def get_score(self, word):
        cleanword = word.lower()
        return  self.fast_idf(word)
        #return  self.idf(word, self.listofdocs)


    def freq(self, word, doc):
        return doc.count(word)
     
     
    def word_count(self, doc):
        return len(doc)
     
     
    def tf(self, word):
        return 1
        #return (self.freq(word, doc) / float(self.word_count(doc)))
     
     
    def num_docs_containing(self, word, list_of_docs):
        count = 0
        for document in list_of_docs:
            if self.freq(word, document) > 0:
                count += 1
        return count + 0.5 # add-alpha smoothing for OOV words
     
    
    def fast_idf( self, word ):
        docfreq = self.docfreq_dict.get( word, 0.5 )
        return math.log( self.numdocs / float( docfreq ) )

    def idf(self, word, list_of_docs):
        docfreq = self.num_docs_containing( word, list_of_docs )
        if docfreq == 0:
            docfreq += 0.5
            
            #print word
            #import pdb
            #pdb.set_trace()

        return math.log(len(list_of_docs) / float( docfreq ) )

def unit_test():
    tfidf = Tfidf({"1": {"name":"restaurant and coffee"}, "2": {"name": "restaurant"}, "3": {"name": "restaurant"}}, {"1": {"name":"starbucks"}, "2": {"name": "restaurant"}, "3": {"name": "restaurant"} }, "name" )
    name1 = "starbucks restaurant"
    name2 = "restaurant"
    name3 = "starbucks"
    name4 = "candy restaurant"
    name5 = "candy ristorante"
    #import pdb
    #pdb.set_trace()

    if name1 == "":
        set1 = set()
    else:
        set1 = set(name4.lower().split())
    if name2 == "":
        set2 = set()
    else:
        set2 = set(name5.lower().split())

    i = list(set1.intersection(set2))
    u = list(set1.union(set2))


    #compute tfidf score
    iscore = sum([tfidf.get_score(word) for word in i])
    uscore = sum([tfidf.get_score(word) for word in u])

    print i,u
    print iscore, uscore

    #pdb.set_trace()
    outcome = 0 if uscore == 0 else float(iscore) / uscore
    print "score", outcome

if __name__ == '__main__':
    unit_test()
