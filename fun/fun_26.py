# 2-gramms

import pandas as pd
import numpy as np
df = pd.read_pickle('msg.pickle')
df = df[df['sender_name'] == 'Вы']
df = df[df['chat_name'] != 'Искандер Гутенберг']
chars = {''}
le = 0
for i in df['text'].tolist():
    i = str(i)
    if i is not None:
        k = i.split()
        for j in range(len(k)-2):
            if j is not None:
                chars.add(k[j] + ' ' + k[j+1])
                le += 1
print(len(df), 'сообщение')
print(le, '2-грамм')
chars = sorted(list(chars))
char_to_num = []

v = 0
for i in chars:
    char_to_num.append((i,0))
    v += 1
v = v-1
char_to_num = dict(char_to_num)

for i in df['text'].tolist():
    i = str(i)
    if i is not None:
        k = i.split()
        for j in range(len(k) - 2):
            if j is not None:
                char_to_num[k[j] + ' ' + k[j+1]] += 1
b = []
a = {k: v for k, v in sorted(char_to_num.items(), key=lambda item: item[1])}
for i, j in list(a.items()):
    b.append((i,j))
b.reverse()
v = 1
for i in b[:20000]:
    print(v, i)
    v +=1