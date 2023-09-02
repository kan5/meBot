# symbols

import pandas as pd

df = pd.read_pickle('msg.pickle')
print(len(df))
# df = df[df['sender_name'] == 'Вы']
# df = df[df['chat_name'] != 'Искандер Гутенберг']
# df = df[df['is_private'] == 1]
a = []
for i in df['text'].tolist():
    a.append(len(str(i)))

print(a)