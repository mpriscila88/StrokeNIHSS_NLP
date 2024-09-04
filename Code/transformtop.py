

def transform_top_features(df, top_features, feature):
    
    #------------------------------------------------------------------------
    # Transform data with Top features 
    #------------------------------------------------------------------------

   top_features = list(top_features.Features.values)

   from nltk import word_tokenize

   def top_words(words):
       new_word = []
       for word in words:
           if word in top_features:
               new_word.append(word)
       return new_word

   df[feature] = df[feature].apply(lambda x: word_tokenize(x))
   df[feature] = df[feature].apply(lambda x: top_words(x)) 
   df[feature] = df[feature].apply(lambda x: ' '.join((x)))
    
   import re
   df['Ntokens'] = df[feature].astype('str').apply(lambda x: len(re.findall(r'\w+',x)))
    
   return df
    