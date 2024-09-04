

def vectorization_(X_train, X_test):
    
    #------------------------------------------------------------------------
    # Vectorization
    #------------------------------------------------------------------------
    
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    
    # word level tf-idf
    tfidf_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=None, ngram_range=(1,1))
    tfidf_vect.fit(X_train)
    xtrain_tfidf =  tfidf_vect.transform(X_train)
    xvalid_tfidf =  tfidf_vect.transform(X_test)
     
    # Train data
    x_train = pd.DataFrame(xtrain_tfidf.todense(), columns = tfidf_vect.get_feature_names())
    
    # Test data
    x_test = pd.DataFrame(xvalid_tfidf.todense(), columns = tfidf_vect.get_feature_names())
    
    return x_train, x_test, xtrain_tfidf, xvalid_tfidf, tfidf_vect

  