import pandas as pd
import numpy as np
white_list = 'qwertyuiopasdfghjklzxcvbnmйцукенгшщзфывапролдячсмить'
df = pd.read_pickle('msg.pickle')

q = []
a = []
c = len(df)
strk = ''
old_is_i = False
old_who = ''
f = open('history.txt', 'w')
for i in range(c):
    curr = df.at[c]['text']
    who = df.at[c]['chat_name']
    is_i = False
    if df.at[c]['sender_name'] == 'Вы':
        is_i = True
    if old_who == who:
        if old_is_i == is_i:
            strk = strk + ',' + curr
        else:
            f.write(strk + '\n')
            strk = curr
            old_is_i = is_i
    else:
        if old_is_i and is_i:
            f.write(strk + '\n')
            strk = ''
