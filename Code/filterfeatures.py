
def filter_features(X_train, outcome, feature, path):
    
    import pandas as pd
    import re
    import sys
    sys.path.insert(0, path) # insert path
    from vectorization import vectorization_
    from transformtop import transform_top_features

    #vectorization
    x_train, x_train, xtrain_tfidf, xtrain_tfidf, tfidf_vect = vectorization_(X_train[feature], X_train[feature])
    
    #matrix vector of features
    matrix_features = pd.DataFrame(xtrain_tfidf.todense(), columns = tfidf_vect.get_feature_names())
      
    #sparsity
    a = (matrix_features == 0).astype(int).sum(axis=0)/len(matrix_features)*100
    a = a[a<90]
    a = pd.DataFrame(a.index,columns = ['Features'])
          
    matrix_features = matrix_features[a.Features]
    
    top_features = pd.DataFrame(list(matrix_features.columns.values), columns={'Features'})
     
    X_train = transform_top_features(pd.DataFrame(X_train), top_features, feature)
    
    # Remove duplicated words in row
    X_train.Notes = X_train.Notes.astype(str).apply(lambda x: re.sub(r'\b(\w+)( \1\b)+', r'\1', x))

    # Remove duplicated spaces
    X_train.Notes = X_train.Notes.apply(lambda x: " ".join(x.split())) 

    return X_train