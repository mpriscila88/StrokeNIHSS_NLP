
def remove_stop_words_(n):
    
    remove = [
              #pronouns      
              'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
              'ourselves', 'you', 'youre', 'youve', 'youll',
              'youd', 'your', 'yours', 'yourself', 'yourselves',
              'he', 'him', 'his', 'himself','she', 'shes', 
              'her', 'hers', 'herself', 'it', 'its', 
              'itself', 'they', 'them', 'their', 'theirs', 
              'themselves',
             
              #verbs
              "am", "is", "are", "was", "were", "be", "been", "being", 
              "have", "has", "had", "having", "do", "does", "did", "doing",
              'go','went','will', "can", 'sent',
               
              #people
              'who', "whom", 'mr', 'dr','dear', 'resident', 'physician','mrn', 
              'father', 'mother', 'provider', 'ehr', 'rn','clinician',
              'md','inpatient', 'patient', 'sister','brother','partner','ros',
              'husband','wife','spouse','person','staff', 'name','pcp','dob',
              'medic','doct','daughter', 'phd','pgy','family','hcp','nurse','np',
              'partners','patients', 'pta', 'slp', 'therapist', 'son',
                 
              
              #prepositions etc
              'if','of', 'ac','in', 'out', 'by', 'at', 'fo', 'nan','hea','pro',
              'as', 'or', "a", 'an', 'for','the', 'with', 'to','be','from',
              'about','should','would','could','same','thank','please',
              'another', 'either','every', 'although', 'but', 'yet',
              'this','that','what','there', 'here',"these", "those", 'also',
              'as well', 'too', "because",  "until", "while",'when', "where", 
              "why", "how",'then','throughout',"against", "between", "into", 
              "through","during","before","after", "once", "few", "more",
              'ever', "on", "off", "over", "under", "again",
              "further", 'and', 'which', 'yes','other','worst','best',"most", 
              "some", "such", "only", "own",  "so", "than", "very", 
              'however', 'even', "just", "above","below",
              #'up',"down", "all", "any", "both", "each", 
  
              #words others
              'summary','facesheet','items','code','phone','visit','attend',
              'encounter','note', 'admit', 'admission', 'admitted', 'consult', 
              'file', 'report', 'page', 'pager', 'comment', 'service', 
              'relate', 'send', 'edit','edited', 'document','documentation', 
              'part','use', 'errors', 'set', 'education', 'study', 'find',
              'notify', 'systems', 'review', 'assessment', 'pertinent',
              'contact', 'outside',   'diagnose',  'information', 'index',
              'additional', 'result', 'schedule', 'main', 'former', 'basic', 
              'comments','documented','documents','data','syring',
              
              #abbr others
              'mdd', 'apt','resu', 'con','dis','mch', 'wnl', 'mghe', 'diff',
              'id','hs','hid','post','nt','tid', 'nad', 'pa','na', 'msk','cc',
              're', 'nih','score','cmf','vs',
  
              #location
              'massachusetts','ma','address','newton','highland', 'icu',
              'boston', 'fa', 'street','waltham','sommerville',
              'avenue','mghw','cambridge','revere','charlestown','wa',
              'barrasso','salem','lincoln', 'luckhurst','st','chelsea',
              'mghg','nc','mccann','mgh','bwh','webster','lynn','haverhill',
              'bi', 'needham', 'massachusetts', 'general', 'hospital', 'mgh',  
              'pgy2', 'department', 'ellison', 'neurology', 'brigham', 'women',
              'logan', 'airport', 'lunder','blake','winthrop','wi','place',
              'ave','cmf',
              
              #time, order, numbers
              'date', "first","second","third","fourth","fifth","sixth", 
              "seventh","eigth","nineth","tenth", 'january','february',
              'march','april','may','june', 'july','august','september',
              'october','november','december',
              'monday','tuesday','wednesday','thursday','friday','saturday',
              'sunday','winter','spring','summer','autumn', #'one', 'two', 
              # 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
              'am', 'pm', 'hour', 'hours', 'hr', 'hrs', 'day', 'days', 'minutes',
              'year', 'years', 'week', 'weeks', 'months', 'month', 'today', 
              'currently','yesterday', 'yo', 'hourly', 'nightly',
              'daily', 'monthly', 'yearly', 'weekly', 'last',"now",
              
              #units|frequency
              'mg', 'ml','mmhg', 'g', 'cm', 'ii', 'xii', 'lb', 'xl', 'kg','xl',
              'pmh','unit', 'units', 'qh', 'qd','per','sig','bid','ml', 'mg',
              'mm', 'mcg', 'dl', 'mol', 'mmol', 'neuts', 'oz', 'ng', 'qhs', 'qd', 'pf',
              
              #labs|vitals
              'wbc', 'hgb', 'rdw', 'rbc', 'cbc', 'plt',  'ldl', 'gtt', 'bun',
              'hdl', 'glu', 'lipid','spo','lfts', 'hld', 'hbac', 'hct','ca', 'cr',
              'creatinine', 'hb', 'hemoglobin', 'phos', 'potassium', 'magnesium',
              'phosphate', 'calcium', 'urea', 'zinc','mcv','mch','mchc',
              'sodium','chloride','ammonia','glucose', 'calc',
              'monos', 'eos', 'baso', 'basos', 'neutrophils', 'troponin', 
              'cholesterol', 'triglycerides', 'mpv',  'lymph', 
              'lymphs', 'mono', 'neutrophil', 'chol', 'nrbc', 'gfr', 'ua',
              'cl','tp','co', 'bun', 'cre', 'pt', 'inr', 'lact', 'tropt',
              'sgpt', 'sgot', 'ntbnp', 'alkp', 'tbili', 'dbili', 'alb', 'tsh',
              'take',  'tab', 'tablet', 'tablets',
              'vitals', 'lab', 'labs',


              #alphabet
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', #'l', 
              'm', 'n', 'o', 'p', 'q', #'r', 
              's', 't','u', 'v', 'w', 'x', 'y', 'z']
    
    from nltk.tokenize import word_tokenize 

    def word(x, remove):
        x = word_tokenize(x)
        filtered_sentence = []
        for w in x: 
            if (w not in remove) & (len(w)>=1): # len(w)>=1
                filtered_sentence.append(w) 
        filtered_sentence = ' '.join(filtered_sentence)        
        return filtered_sentence

    n.Notes  = n.Notes.astype(str).apply(lambda x: word(x, remove))
    
    
    return n