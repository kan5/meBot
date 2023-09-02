# words
import pandas as pd
df = pd.read_pickle('msg.pickle')
df = df[df['is_private'] == 1]
df = df[df['is_event'] == 0]
df = df[df['is_club'] == 0]
df = df[df['is_conversation'] == 0]

chars = {'привет': 0}
le = 0
for i in df['text'].tolist():
    i = str(i).lower()
    if i is not None:
        k = i.split()
        for j in k:
            if j is not None:
                try:
                    chars[j] += 1
                except:
                    chars[j] = 1
                le += 1

a = {k: v for k, v in sorted(chars.items(), key=lambda item: item[1], reverse=True)}

print(a)
proc = 0.975 * le
c = 0
k = 0
t = 0

st_voc = []
for i, b in a.items():
    if c < proc:
        c += b
        k += 1
        st_voc.append(i)
    t += 1
print(len(st_voc))
# %text / %words
print(k/t)

vocab = []
f = open('vocab.txt', 'r', encoding='utf8')
for i in f.readlines():
    i = i[:-1]
    i.replace('#', '')
    vocab.append(i)
f.close()
chars = []
kse = {'привет'}
ase = {'привет'}
k = 0
a = 0
for i in df['text'].tolist():
    i = str(i).lower()
    if i is not None:
        for j in i.split():
            if j is not None:
                if j in st_voc:
                    ase.add(j)
                    if j in vocab:
                        kse.add(j)
# words in embedding / my words
print(len(kse)/len(ase))
# 13 00 883 wagen

