import pandas as pd
import numpy as np
np.random.seed(91)
import time



start_time = time.time()

df = pd.read_pickle('msg.pickle')
# df = df[df['is_private'] == 1]
print(len(df))

# text handler
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

chars = {k: v for k, v in sorted(chars.items(), key=lambda item: item[1], reverse=True)}
a = chars.copy()
text_dict_size = 0
for i in a.keys():
    if text_dict_size < 260:
        chars[i] = text_dict_size
        text_dict_size += 1
    else:
        chars.pop(i)
a = None


def text_encode(strok):
    a = chars.keys()
    sp = []
    c = 0
    for i in strok:
        if i in a:
            sp.append(i)
    return sp


def text_decode(sp):
    for i in range(len(sp)):
        for j, c in chars.items():
            if sp[i] == c:
                sp[i] = j
                break
    return ''.join(sp)


# chat name
# sender name

chat_names = dict([])
sender_names = dict([])
chat_names['New Interlocutor0123'] = 0
sender_names['New Interlocutor0123'] = 0
for i in df['chat_name']:
    chat_names[i] = 0
for i in df['sender_name']:
    sender_names[i] = 0
chat_names_dict_size = 0
sender_names_dict_size = 0
for i in chat_names.keys():
    chat_names[i] = chat_names_dict_size
    chat_names_dict_size += 1
for i in sender_names.keys():
    sender_names[i] = sender_names_dict_size
    sender_names_dict_size += 1


def chat_name_encode(strok):
    b = np.zeros(chat_names_dict_size, dtype='float32')
    b[chat_names[strok]] = 1.
    return b


def sender_name_encode(strok):
    b = np.zeros(sender_names_dict_size, dtype='float32')
    b[sender_names[strok]] = 1.
    return b


def chat_name_decode(be):
    a = chat_names_dict_size
    for j in range(a):
        if be[j] == 1.:
            sp = j
            break
    for j, c in chat_names.items():
        if sp == c:
            sp = j
            break
    return sp


def sender_name_decode(be):
    a = sender_names_dict_size
    for j in range(a):
        if be[j] == 1.:
            sp = j
            break
    for j, c in sender_names.items():
        if sp == c:
            sp = j
            break
    return sp


size_of_data_input = 1000
input_texts = [text_encode(i) for i in df[:size_of_data_input]['text']]
input_chat_names = [chat_name_encode(i) for i in df[:size_of_data_input]['chat_name']]
input_sender_names = [sender_name_encode(i) for i in df[:size_of_data_input]['sender_name']]
input_dates = [np.array([i], dtype='int64') for i in df[:size_of_data_input]['date']]
input_chat_type = []
for i in df[:size_of_data_input][['is_private', 'is_event', 'is_conversation', 'is_club']].values:
    input_chat_type.append(np.array(i, dtype='float32'))
input_is_redacted = [np.ones(1, dtype='float32') if i == 1 else np.zeros(1, dtype='float32')
                     for i in df[:size_of_data_input]['is_redacted']]
size_of_dense_part = len(input_chat_names[0]) + len(input_sender_names[0]) \
                    + len(input_chat_type[0]) + len(input_is_redacted[0])
input_common = np.array([np.zeros(size_of_dense_part)])
for i in range(size_of_data_input):
    input_common = np.vstack((input_common, np.hstack((input_chat_names[i], input_sender_names[i],
                                                       input_chat_type[i], input_is_redacted[i]))))
input_common = np.delete(input_common, (0), axis=0)

from keras.models import Model
from keras.layers import LSTM, TimeDistributed, Dense, Embedding
from keras import Input


dense_input = Input(shape=(None,), dtype='float32', name='common')
date_input = Input(shape=(None,), dtype='int64', name='date')



encoder_inputs = Input(shape=(MAX_LEN,), dtype='int32', )
encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
encoder_outputs, state_h, state_c = encoder_LSTM(encoder_inputs)

decoder_inputs = Input(shape=(MAX_LEN,), dtype='int32', )
decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])

# dense_layer = Dense(VOCAB_SIZE, activation='softmax')
outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], outputs)

model = Model([text_input, dense_input, date_input], answer)


print("--- %s seconds ---" % (time.time() - start_time))
