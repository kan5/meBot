# symbols

import pandas as pd

df = pd.read_pickle('msg.pickle')

df = df[df['sender_name'] == 'Вы']
# df = df[df['chat_name'] != 'Искандер Гутенберг']
# df = df[df['is_private'] == 1]
le = 0
chars = {' ': 0}
for i in df['text'].tolist():
    i = str(i)
    if i is not None:
        for j in i:
            if j is not None:
                try:
                    chars[j] += 1
                except:
                    chars[j] = 1
                le += 1
print(le, 'символ')


chars = {k: v for k, v in sorted(chars.items(), key=lambda item: item[1], reverse=True)}
v = 1
for i in chars.items():
    print(i, v)
    v += 1
print(len(chars))