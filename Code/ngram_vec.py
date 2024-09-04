
def ngram(X_train, X_test, feature, ngram_range):
    #------------------------------------------------------------------------
    # Vectorization
    #------------------------------------------------------------------------
        
    import pandas as pd
    import re
    from sklearn.feature_extraction.text import CountVectorizer
    
    # word level tf-idf
    
    tfidf_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=None, ngram_range=ngram_range)
    
    #tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=None, ngram_range=ngram_range)
    tfidf_vect.fit(X_train[feature])
    xtrain_tfidf =  tfidf_vect.transform(X_train[feature])
    xvalid_tfidf =  tfidf_vect.transform(X_test[feature])  
  
    xtrain_tfidf2 = xtrain_tfidf.astype('uint8')
    
    # Train data
    x_train = pd.DataFrame(xtrain_tfidf2.todense(), columns = tfidf_vect.get_feature_names())
    #x_train = pd.DataFrame(xtrain_tfidf.todense(), columns = tfidf_vect.get_feature_names())
    x_test = pd.DataFrame(xvalid_tfidf.todense(), columns = tfidf_vect.get_feature_names())
    
    #sparsity
    a = (x_train == 0).astype(int).sum(axis=0)/len(x_train)*100
  
    #a.hist()
    a = a[a<90]
    a = pd.DataFrame(a.index,columns = ['Features'])
  
    x_train = x_train[a.Features]
    x_test = x_test[a.Features]

    return x_train, x_test

