import re

def regex_extract(text):
    results = set()

    for m in re.finditer(r'(?:[A-Z][a-z]{2,40}(?:\s+(?:of|in|s|for|and|the|up|to|on))?(?:\s+[A-Z][a-z]{2,40})){2,}', text):
        results.add(m.group().strip())

    for m in re.finditer(r'(?:[A-Z][A-Za-z]{2,20}\s){1,4}(?:Data(?:base|set)?|Survey|Registry|Study|Programme|Initiative)\b'
        ,text):
        results.add(m.group().strip())

    pattern_3 = r'''
    \b([A-Z]{3,5})\b
    (?=[^.\n]{0,50}(?:Data(?:base|set)?|Survey|Registry|Study|Programme|Initiative)\b)
    |
    (?:(?:Data(?:base|set)?|Survey|Registry|Study|Programme|Initiative)\b[^.\n]{0,50})
    \b([A-Z]{3,5})\b
    '''

    for m in re.finditer(pattern_3, text):
        results.add(m.group().strip())

    return set(r for r in results)