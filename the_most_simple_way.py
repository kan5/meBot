import pandas as pd
import pickle
import numpy as np
import time

start_time = time.time()


def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def set_set():

    white_list = 'qwertyuiopasdfghjklzxcvbnmйцукенгшщзфывапролдячсмить'
    black_list = ['фотография', 'аудиозапись', 'запись на стене', 'прикреплённое сообщение', 'стикер', 'вышел из беседы', 'прикреплённых сообщения', 'вернулся в беседу',]

    df = pd.read_pickle('msg1.pickle')
    q = []
    a = []
    c = len(df)
    i = c - 1
    long_list = 0
    old_chat = df.at[c - 1, 'chat_name']
    is_i = True
    while i > 0:
        new_chat = df.at[i, 'chat_name']
        is_i = df.at[i, 'sender_name'] == 'Вы'
        if old_chat == new_chat and is_i and df.at[i - 1, 'sender_name'] != 'Вы':
            qn = df.at[i - 1, 'text'].lower()
            an = df.at[i, 'text'].lower()
            fl = True
            for j in black_list:
                if j in qn or j in an:
                    fl = False
                    break
            if fl:
                q.append(qn)
                a.append(an)
        old_chat = new_chat
        i -= 1
    print('pairs:', len(q))

    chars = {'привет': 1}
    le = 0
    for i in q:
        if i is not None:
            for j in i.split():
                try:
                    chars[j] += 1
                except:
                    chars[j] = 1
                le += 1

    chars['<UNK>'] = 0
    chars = {k: v for k, v in sorted(chars.items(), key=lambda item: item[1], reverse=True)}

    v = len(chars)-1
    for i, j in chars.items():
        chars[i] = v
        v -= 1

    wrd2x = chars
    x2wrd = {j: i for i, j in wrd2x.items()}

    qn = []
    an = []
    c = len(a)
    for i in range(c):
        qt = []
        for j in q[i]:
            for k, t in wrd2x.items():
                if k == j:
                    qt.append(t)
                    break
        qn.append(qt)
    ans2x = {a[0]: 0}
    x2ans = {0: a[0]}
    for i in a:
        ans2x[i] = 1
    c = 0
    for i, j in ans2x.items():
        ans2x[i] = c
        x2ans[c] = i
        c += 1
    for i in a:
        an.append(ans2x[i])


    save_obj(qn, 'questions')
    save_obj(an, 'answers')
    save_obj(wrd2x, 'wrd2x')
    save_obj(x2wrd, 'x2wrd')
    save_obj(ans2x, 'ans2x')
    save_obj(x2ans, 'x2ans')

def build_model(count_of_words, types_of_answers):
    from keras import models
    from keras import layers

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(count_of_words, )))
    model.add(layers.Dense(types_of_answers, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def vectorize_in(seq, words):
    lseq = len(seq)
    try:
        fl = True
        seq[0][0] == 1
    except:
        fl = False
    if fl:
        result = np.zeros((lseq, words))
        for i in range(lseq):
            for j in seq[i]:
                result[i, j] = 1.
    else:
        result = np.zeros((1, words ))
        for i in seq:
            result[0, i] = 1.
    return result


def vectorize_out(answers, count_types):
    lans = len(answers)
    result = np.zeros((lans, count_types))
    for i in range(lans):
        result[i] = answers[i]
    return result


def interface(input, model, w2x, x2a):
    q = []
    for i in input.lower().split():
        for j, k in w2x.items():
            if j == i:
                q.append(k)
    if q == []:
        return ''
    b = vectorize_in(q, len(w2x))
    a = model.predict(b)
    c = len(a)
    max = 0
    maxid = 0
    for i in range(c):
        if a[0, i] > max:
            max = a[0, i]
            maxid = i
    return x2a[maxid]

import os

if not os.path.isdir('obj'):
    os.mkdir('obj')
if not os.path.exists('obj/x2ans.pkl'):
    set_set()

x2wrd = load_obj('x2wrd')
wrd2x = load_obj('wrd2x')
qn = load_obj('questions')
an = load_obj('answers')
ans2x = load_obj('ans2x')
x2ans = load_obj('x2ans')
# print(x2wrd)
# print(wrd2x)
print(qn)
print(an)
# print(ans2x)
# print(x2ans)

model = build_model(len(wrd2x), len(ans2x))

questions = vectorize_in(qn, len(wrd2x))
answers = vectorize_out(an, len(ans2x))

# model.load_weights('mod.h5')
history = model.fit(questions, answers, batch_size=64, epochs=10, validation_split=0.1, shuffle=True)
model.save_weights('mod.h5')

# import matplotlib.pyplot as plt
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()


print(interface('Првет, крошка, ка жизнь?', model, wrd2x, x2ans))
print(interface('Кто ты?', model, wrd2x, x2ans))
print(interface('Ты программист?', model, wrd2x, x2ans))
print(interface('просто', model, wrd2x, x2ans))


print("--- %s seconds ---" % (time.time() - start_time))