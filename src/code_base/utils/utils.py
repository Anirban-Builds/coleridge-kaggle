import re

def clean_text(txt, Lower=False):
    '''
    Function for cleaning text and preprocessing
    '''
    txt =  re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()
    return txt.lower() if Lower else txt