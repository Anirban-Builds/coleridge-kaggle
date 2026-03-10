import re

def clean_text(txt):
    '''
    Function for cleaning text and preprocessing 
    '''
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()