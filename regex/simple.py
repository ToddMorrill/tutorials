# https://www.youtube.com/watch?v=K8L6KVGG-7o

import re

text_to_search = '''
abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890
Ha HaHa
MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] \ | ( )
coreyms.com
321-555-4321
123.555.1234
123*555*1234
800-555-1234
900-555-1234
Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T
'''

urls = '''
https://www.google.com
http://coreyms.com
https://youtube.com
https://www.nasa.gov
'''

def example_1():
    sentence = 'Start a sentence and then bring it to an end'

    pattern = re.compile(r'start', re.I)

    matches = pattern.search(sentence)

    print(matches)

def regex_search(pat, string, *args):
    pattern = re.compile(pat, *args)

    matches = pattern.finditer(string)
    for m in matches:
        print(m)

def regex_groups(pat, string, group=None, *args):
    pattern = re.compile(pat, *args)

    matches = pattern.finditer(string)
    for m in matches:
        if group:
            print(m.group(group))
        else:
            print(m)

def regex_sub(pat, string, groups=None, *args):
    pattern = re.compile(pat, *args)
    print(pattern.sub(groups, string))

def regex_findall(pat, string, *args):
    pattern = re.compile(pat, *args)

    matches = pattern.findall(string)
    for m in matches:
        print(m)
# regex_search(r'abc', text_to_search)
# regex_search(r'abc', text_to_search, re.I)
# regex_search(r'\.', text_to_search)
# regex_search(r'coreyms\.com', text_to_search)

# with open('data.txt', 'r') as f:
#     data = f.read()

# regex_search(r'\d\d\d.\d\d\d.\d\d\d\d', data)
# regex_groups(r'https?://(www\.)?(\w+)(\.\w+)', text_to_search, 3)

# regex_sub(r'https?://(www\.)?(\w+)(\.\w+)', urls, r'\2\3')

# only prints first group
regex_findall(r'M(r|s|rs)\.?\s[A-Z]\w*', text_to_search)

# re.match matches patterns at the start of the string and returns the match, else None
# re.search matches the first occurrence of the string and returns the match, else None