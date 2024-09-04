
import sys
import os
import pandas as pd
import numpy as np
import re
from dateutil.relativedelta import relativedelta


def n2w(n):
    
    def sub_(n, e1, e2): 
        n.Notes = n.Notes.astype(str).apply(lambda x: re.sub(e1,e2,x)) 
        return n

    #LOC alert
    n = sub_(n,'alert\s?0\s?\(0-3\)','alert keenly responsive')
    n = sub_(n,'alert\s?1\s?\(0-3\)','arouses to minor stimulation')
    n = sub_(n,'alert\s?2\s?\(0-3\)','movements to pain')
    n = sub_(n,'alert\s?3\s?\(0-3\)','unresponsive')

    n = sub_(n,'1a- 0','alert keenly responsive')
    n = sub_(n,'1a- 1','arouses to minor stimulation')
    n = sub_(n,'1a- 2','movements to pain')
    n = sub_(n,'1a- 3','unresponsive')
      
    n = sub_(n,'0 1\s?b','alert keenly responsive')
    n = sub_(n,'1 1\s?b','arouses to minor stimulation')
    n = sub_(n,'2 1\s?b','movements to pain')
    n = sub_(n,'3 1\s?b','unresponsive')
    
 
    # LOC question
    n = sub_(n,'question\s?0\s?\(0-2\)','answers both questions correctly')
    n = sub_(n,'question\s?1\s?\(0-2\)','dysarthric')
    n = sub_(n,'question\s?2\s?\(0-2\)','aphasic')
    
    n = sub_(n,'1b- 0','answers both questions correctly')
    n = sub_(n,'1b- 1','dysarthric')
    n = sub_(n,'1b- 2','aphasic')
     
    n = sub_(n,'0 1\s?c','answers both questions correctly')
    n = sub_(n,'1 1\s?c','dysarthric')
    n = sub_(n,'2 1\s?c','aphasic')
    
    # LOC commands
    n = sub_(n,'commands\s?0\s?\(0-2\)','performs both tasks correctly')
    n = sub_(n,'commands\s?1\s?\(0-2\)','performs one task')
    n = sub_(n,'commands\s?2\s?\(0-2\)','not perform tasks')
    
    n = sub_(n,'1c- 0','performs both tasks correctly')
    n = sub_(n,'1c- 1','performs one task')
    n = sub_(n,'1c- 2','not perform tasks')

    n = sub_(n,'0 2 (best)?\s?gaze','performs both tasks correctly')
    n = sub_(n,'1 2 (best)?\s?gaze','performs one task')
    n = sub_(n,'2 2 (best)?\s?gaze','not perform tasks')

    
    # Horizontal gaze
    n = sub_(n,'gaze\s?0\s?\(0-2\)','gaze normal')
    n = sub_(n,'gaze\s?1\s?\(0-2\)','partial gaze palsy')
    n = sub_(n,'gaze\s?2\s?\(0-2\)','forced gaze palsy')
    
    n = sub_(n,'gaze 2- 0','gaze normal')
    n = sub_(n,'gaze 2- 1','partial gaze palsy')
    n = sub_(n,'gaze 2- 2','forced gaze palsy')
    
    n = sub_(n,'0 3 visual fields?','gaze normal')
    n = sub_(n,'1 3 visual fields?','partial gaze palsy')
    n = sub_(n,'2 3 visual fields?','forced gaze palsy')
    
    # Visual Field 
    n = sub_(n,'visual field\s?0\s?\(0-3\)','no visual loss')
    n = sub_(n,'visual field\s?1\s?\(0-3\)','partial hemianopia')
    n = sub_(n,'visual field\s?2\s?\(0-3\)','complete hemianopia')
    n = sub_(n,'visual field\s?3\s?\(0-3\)','bilateral hemianopia')
    
    n = sub_(n,'visual (field)?\s?3- 0','no visual loss')
    n = sub_(n,'visual (field)?\s?3- 1','partial hemianopia')
    n = sub_(n,'visual (field)?\s?3- 2','complete hemianopia')
    n = sub_(n,'visual (field)?\s?3- 3','bilateral hemianopia')
    
    n = sub_(n,'0 4 facial palsy','no visual loss')
    n = sub_(n,'1 4 facial palsy','partial hemianopia')
    n = sub_(n,'2 4 facial palsy','complete hemianopia')
    n = sub_(n,'3 4 facial palsy','bilateral hemianopia')
    
    
    # Facial Palsy 
    n = sub_(n,'facial palsy\s?0\s?\(0-3\)','normal symmetrical movements')
    n = sub_(n,'facial palsy\s?1\s?\(0-3\)','minor paralysis')
    n = sub_(n,'facial palsy\s?2\s?\(0-3\)','partial paralysis')
    n = sub_(n,'facial palsy\s?3\s?\(0-3\)','complete paralysis')
    
    n = sub_(n,'facial palsy 4- 0','normal symmetrical movements')
    n = sub_(n,'facial palsy 4- 1','minor paralysis')
    n = sub_(n,'facial palsy 4- 2','partial paralysis')
    n = sub_(n,'facial palsy 4- 3','complete paralysis')

    n = sub_(n,'0 5\s?a','normal symmetrical movements')
    n = sub_(n,'1 5\s?a','minor paralysis')
    n = sub_(n,'2 5\s?a','partial paralysis')
    n = sub_(n,'3 5\s?a','complete paralysis')
    
 
    # Motor L arm
    n = sub_(n,'motor left arm\s?0\s?\(0-4\)','mlarm no drift')
    n = sub_(n,'motor left arm\s?1\s?\(0-4\)','mlarm drift')
    n = sub_(n,'motor left arm\s?2\s?\(0-4\)','mlarm drift hits bed')
    n = sub_(n,'motor left arm\s?3\s?\(0-4\)','mlarm no effort')
    n = sub_(n,'motor left arm\s?4\s?\(0-4\)','mlarm no movement')
    
    n = sub_(n,'motor left arm 5a- 0','mlarm no drift')
    n = sub_(n,'motor left arm 5a- 1','mlarm drift')
    n = sub_(n,'motor left arm 5a- 2','mlarm drift hits bed')
    n = sub_(n,'motor left arm 5a- 3','mlarm no effort')
    n = sub_(n,'motor left arm 5a- 4','mlarm no movement')
    
    n = sub_(n,'0 5\s?b','mlarm no drift')
    n = sub_(n,'1 5\s?b','mlarm drift')
    n = sub_(n,'2 5\s?b','mlarm drift hits bed')
    n = sub_(n,'3 5\s?b','mlarm no effort')
    n = sub_(n,'4 5\s?b','mlarm no movement')
    
    # Motor R arm
    n = sub_(n,'motor right arm\s?0\s?\(0-4\)','mrarm no drift')
    n = sub_(n,'motor right arm\s?1\s?\(0-4\)','mrarm drift')
    n = sub_(n,'motor right arm\s?2\s?\(0-4\)','mrarm drift hits bed')
    n = sub_(n,'motor right arm\s?3\s?\(0-4\)','mrarm no effort')
    n = sub_(n,'motor right arm\s?4\s?\(0-4\)','mrarm no movement')
    
    n = sub_(n,'motor right arm 5b- 0','mrarm no drift')
    n = sub_(n,'motor right arm 5b- 1','mrarm drift')
    n = sub_(n,'motor right arm 5b- 2','mrarm drift hits bed')
    n = sub_(n,'motor right arm 5b- 3','mrarm no effort')
    n = sub_(n,'motor right arm 5b- 4','mrarm no movement')
    
    n = sub_(n,'0 6\s?a','mrarm no drift')
    n = sub_(n,'1 6\s?a','mrarm drift')
    n = sub_(n,'2 6\s?a','mrarm drift hits bed')
    n = sub_(n,'3 6\s?a','mrarm no effort')
    n = sub_(n,'4 6\s?a','mrarm no movement')
    
    # Motor L leg
    n = sub_(n,'motor left leg\s?0\s?\(0-4\)','mlleg no drift')
    n = sub_(n,'motor left leg\s?1\s?\(0-4\)','mlleg drift')
    n = sub_(n,'motor left leg\s?2\s?\(0-4\)','mlleg drift hits bed')
    n = sub_(n,'motor left leg\s?3\s?\(0-4\)','mlleg no effort')
    n = sub_(n,'motor left leg\s?4\s?\(0-4\)','mlleg no movement')
    
    n = sub_(n,'motor left leg 6a- 0','mlleg no drift')
    n = sub_(n,'motor left leg 6a- 1','mlleg drift')
    n = sub_(n,'motor left leg 6a- 2','mlleg drift hits bed')
    n = sub_(n,'motor left leg 6a- 3','mlleg no effort')
    n = sub_(n,'motor left leg 6a- 4','mlleg no movement')

    n = sub_(n,'0 6\s?b','mlleg no drift')
    n = sub_(n,'1 6\s?b','mlleg drift')
    n = sub_(n,'2 6\s?b','mlleg drift hits bed')
    n = sub_(n,'3 6\s?b','mlleg no effort')
    n = sub_(n,'4 6\s?b','mlleg no movement')
    
    # Motor R leg
    n = sub_(n,'motor right leg\s?0\s?\(0-4\)','mrleg no drift')
    n = sub_(n,'motor right leg\s?1\s?\(0-4\)','mrleg drift')
    n = sub_(n,'motor right leg\s?2\s?\(0-4\)','mrleg drift hits bed')
    n = sub_(n,'motor right leg\s?3\s?\(0-4\)','mrleg no effort')
    n = sub_(n,'motor right leg\s?4\s?\(0-4\)','mrleg no movement')
    
    n = sub_(n,'motor right leg 6b- 0','mrleg no drift')
    n = sub_(n,'motor right leg 6b- 1','mrleg drift')
    n = sub_(n,'motor right leg 6b- 2','mrleg drift hits bed')
    n = sub_(n,'motor right leg 6b- 3','mrleg no effort')
    n = sub_(n,'motor right leg 6b- 4','mrleg no movement')
    
    n = sub_(n,'0 7 limb ataxia','mrleg no drift')
    n = sub_(n,'1 7 limb ataxia','mrleg drift')
    n = sub_(n,'2 7 limb ataxia','mrleg drift hits bed')
    n = sub_(n,'3 7 limb ataxia','mrleg no effort')
    n = sub_(n,'4 7 limb ataxia','mrleg no movement')
    
    # Ataxia
    n = sub_(n,'ataxia\s?0\s?\(0-2\)','no ataxia')
    n = sub_(n,'ataxia\s?1\s?\(0-2\)','ataxia in one limb')
    n = sub_(n,'ataxia\s?2\s?\(0-2\)','ataxia in two limbs')
    
    n = sub_(n,'ataxia 7- 0','no ataxia')
    n = sub_(n,'ataxia 7- 1','ataxia in one limb')
    n = sub_(n,'ataxia 7- 2','ataxia in two limbs')
    
    n = sub_(n,'0 8 sensory','no ataxia')
    n = sub_(n,'1 8 sensory','ataxia in one limb')
    n = sub_(n,'2 sensory','ataxia in two limbs')

    # Sensory
    n = sub_(n,'sensory\s?0\s?\(0-2\)','no sensory loss')
    n = sub_(n,'sensory\s?1\s?\(0-2\)','mild moderate loss')
    n = sub_(n,'sensory\s?2\s?\(0-2\)','no response')
    
    n = sub_(n,'sensory 8- 0','no sensory loss')
    n = sub_(n,'sensory 8- 1','mild moderate loss')
    n = sub_(n,'sensory 8- 2','no response')
    
    n = sub_(n,'0 9 (best)?\s?language','no sensory loss')
    n = sub_(n,'1 9 (best)?\s?language','mild moderate loss')
    n = sub_(n,'2 9 (best)?\s?language','no response')
    
    # Language
    n = sub_(n,'language\s?0\s?\(0-3\)','no aphasia')
    n = sub_(n,'language\s?1\s?\(0-3\)','mild moderate aphasia')
    n = sub_(n,'language\s?2\s?\(0-3\)','severe aphasia')
    n = sub_(n,'language\s?3\s?\(0-3\)','unresponsive')
    
    n = sub_(n,'language 9- 0','no aphasia')
    n = sub_(n,'language 9- 1','mild moderate aphasia')
    n = sub_(n,'language 9- 2','severe aphasia')
    n = sub_(n,'language 9- 3','unresponsive')
    
    n = sub_(n,'0 10 dysarthria','no aphasia')
    n = sub_(n,'1 10 dysarthria','mild moderate aphasia')
    n = sub_(n,'2 10 dysarthria','severe aphasia')
    n = sub_(n,'3 10 dysarthria','unresponsive')
    
    # dysarthria
    n = sub_(n,'dysarthria\s?0\s?\(0-2\)','normal speech')
    n = sub_(n,'dysarthria\s?1\s?\(0-2\)','mild moderate dysarthria')
    n = sub_(n,'dysarthria\s?2\s?\(0-2\)','severe dysarthria')
    
    n = sub_(n,'dysarthria 10- 0','normal speech')
    n = sub_(n,'dysarthria 10- 1','mild moderate dysarthria')
    n = sub_(n,'dysarthria 10- 2','severe dysarthria')
    
    n = sub_(n,'0 11 extinction','normal speech')
    n = sub_(n,'1 11 extinction','mild moderate dysarthria')
    n = sub_(n,'2 11 extinction','severe dysarthria')
    
    # Extinction and Inattention
    n = sub_(n,'inattention\s?0\s?\(0-2\)','no abnormality')
    n = sub_(n,'inattention\s?1\s?\(0-2\)','visual tactile auditory spatial personal inattention')
    n = sub_(n,'inattention\s?2\s?\(0-2\)','profound hemi inattention')

    n = sub_(n,'inattention 11- 0','no abnormality')
    n = sub_(n,'inattention 11- 1','visual tactile auditory spatial personal inattention')
    n = sub_(n,'inattention 11- 2','profound hemi inattention')

    n = sub_(n,'extinction\s?0\s?\(0-2\)','no abnormality')
    n = sub_(n,'extinction\s?1\s?\(0-2\)','visual tactile auditory spatial personal inattention')
    n = sub_(n,'extinction\s?2\s?\(0-2\)','profound hemi inattention')

    n = sub_(n,'extinction 11- 0','no abnormality')
    n = sub_(n,'extinction 11- 1','visual tactile auditory spatial personal inattention')
    n = sub_(n,'extinction 11- 2','profound hemi inattention')
     
    n = sub_(n,'extinction\/?(inattention)? 0','no abnormality')
    n = sub_(n,'extinction\/?(inattention)? 1','visual tactile auditory spatial personal inattention')
    n = sub_(n,'extinction\/?(inattention)? 2','profound hemi inattention')

    return n
