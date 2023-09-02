VOCAB_SIZE = 20000
MAX_LEN = 50
EMBEDDING_DIM = 256
# cause we have limited RAM
num_samples = 512 + 1
dense_size = 3322
EPOCHS = 1 # EPOCHS * loop_in_epoch
loop_in_epoch = 4
batch_size = 128
load_local_weights = True

import pandas as pd
import numpy as np

np.random.seed(91)
import time

start_time = time.time()

df = pd.read_pickle('drive/My Drive/Colab Notebooks/msg.pickle')
# df = df[df['is_private'] == 1]

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


# text handler
text_data = []
bos = "<BOS>"
eos = "<EOS>"
for i in df['text']:
    i = str(i).lower()
    c = i.find('http')
    while c >= 0:
        z = i[c:]
        d = len(i[:c])
        t = [j + d for j in [z.find(' '), z.find(','), z.find('\n')] if j > 0]
        if t != []:
            i = i[:c] + i[min(t) + 1:]
        else:
            i = i[:c]
        c = i.find('http')
    text_data.append(bos + i + eos)

flag = True
encoder_input_text = []
decoder_input_text = []
encoder_input_text.append(text_data[0])
for i in text_data[1:-1]:
    encoder_input_text.append(i)
    decoder_input_text.append(i)
decoder_input_text.append(text_data[-1:])
if len(decoder_input_text) > len(encoder_input_text):
    decoder_input_text = decoder_input_text[:len(encoder_input_text)]
elif len(decoder_input_text) < len(encoder_input_text):
    encoder_input_text = encoder_input_text[:len(decoder_input_text)]

from keras.preprocessing.text import Tokenizer


def vocab_creater(text_lists, VOCAB_SIZE):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(text_lists)
    dictionary = tokenizer.word_index

    word2idx = {}
    idx2word = {}
    for k, v in dictionary.items():
        if v < VOCAB_SIZE:
            word2idx[k] = v
            idx2word[v] = k
        if v >= VOCAB_SIZE - 1:
            continue

    return word2idx, idx2word


word2idx, idx2word = vocab_creater(text_lists=text_data, VOCAB_SIZE=VOCAB_SIZE)


def text2seq(encoder_text, decoder_text, VOCAB_SIZE):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(text_data)
    encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
    decoder_sequences = tokenizer.texts_to_sequences(decoder_text)

    return encoder_sequences, decoder_sequences


encoder_sequences, decoder_sequences = text2seq(encoder_text=encoder_input_text, decoder_text=decoder_input_text,
                                                VOCAB_SIZE=VOCAB_SIZE)

from keras.preprocessing.sequence import pad_sequences


def padding(encoder_sequences, decoder_sequences, MAX_LEN):
    encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post',
                                       truncating='post')
    decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post',
                                       truncating='post')

    return encoder_input_data, decoder_input_data


encoder_input_data, decoder_input_data = padding(encoder_sequences, decoder_sequences, MAX_LEN)
from keras.layers import Embedding


def embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN):
    embedding_layer = Embedding(input_dim=VOCAB_SIZE,
                                output_dim=EMBEDDING_DIM,
                                input_length=MAX_LEN,
                                trainable=True)
    return embedding_layer


embedding_layer = embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN)


def decoder_output_creater(decoder_input_data, num_samples, MAX_LEN, VOCAB_SIZE):
    decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")

    for i, seqs in enumerate(decoder_input_data):
        for j, seq in enumerate(seqs):
            if j > 0:
                decoder_output_data[i][j][seq] = 1.

    return decoder_output_data


from sklearn.model_selection import train_test_split


def data_spliter(encoder_input_data, decoder_input_data, test_size1=0.2, test_size2=0.3):
    en_train, en_test, de_train, de_test = train_test_split(encoder_input_data, decoder_input_data,
                                                            test_size=test_size1)
    en_train, en_val, de_train, de_val = train_test_split(en_train, de_train, test_size=test_size2)

    return en_train, en_val, en_test, de_train, de_val, de_test


en_train, en_val, en_test, de_train, de_val, de_test = data_spliter(encoder_input_data, decoder_input_data)

from keras.models import Model
from keras.layers import GRU, LSTM, TimeDistributed, Dense, Embedding, Concatenate
from keras import Input


def seq2seq_model_builder(HIDDEN_DIM=EMBEDDING_DIM, embed_layer=embedding_layer):
    encoder_inputs = Input(shape=(MAX_LEN,), dtype='int32', )

    encoder_embedding = embed_layer(encoder_inputs)
    encoder_GRU = GRU(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h = encoder_GRU(encoder_embedding)
    dense_input = Input(shape=(dense_size,), dtype='float32', )
    concatenate = Concatenate(axis=-1)([dense_input, state_h])
    dense_out = Dense(HIDDEN_DIM, activation='relu')(concatenate)

    decoder_inputs = Input(shape=(MAX_LEN,), dtype='int32', )
    decoder_embedding = embed_layer(decoder_inputs)
    decoder_GRU = GRU(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, state_h_out = decoder_GRU(decoder_embedding, initial_state=[dense_out])
    chat_name_out = Dense(chat_names_dict_size, activation='softmax')(state_h_out)
    date_out = Dense(2)(state_h_out)

    # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs, dense_input], [outputs, chat_name_out, date_out])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], )

    return model


model = seq2seq_model_builder()
model.summary()

if load_local_weights:
    model.load_weights('drive/My Drive/Colab Notebooks/model.h5')

timing = time.time()
all_seqs = len(encoder_sequences)
for j in range(EPOCHS):
    flag = True
    c = 0
    while flag:
        t = num_samples if c + num_samples < all_seqs else all_seqs - c
        # input_texts = [text_encode(i) for i in df[:size_of_data_input]['text']]
        input_chat_names = []
        output_chat_names = []
        b = chat_name_encode(df.at[c, 'chat_name'])
        input_chat_names.append(b)
        for i in df[c + 1:c + t - 1]['chat_name']:
            input_chat_names.append(chat_name_encode(i))
            output_chat_names.append(chat_name_encode(i))
        b = chat_name_encode(df.at[c + t, 'chat_name'])
        output_chat_names.append(b)
        output_chat_names_tensor = np.array(output_chat_names, dtype='float32')
        input_dates = []
        output_dates = []
        b = df.at[c, 'date']
        input_dates.append(np.array([b / 1000000000, b % 1000], dtype='float32'))
        for i in df[c + 1:c + t - 1]['date']:
            a = np.array([i / 1000000000, i % 1000], dtype='float32')
            input_dates.append(a)
            output_dates.append(a)
        b = df.at[c + t, 'date']
        output_dates.append(np.array([b / 1000000000, b % 1000], dtype='float32'))
        output_dates_tensor = np.array(output_dates, dtype='float32')

        input_sender_names = [sender_name_encode(i) for i in df[c:c + t]['sender_name']]
        input_chat_type = []
        for i in df[c:c + t][['is_private', 'is_event', 'is_conversation', 'is_club']].values:
            input_chat_type.append(np.array(i, dtype='float32'))
        input_is_redacted = [np.ones(1, dtype='float32') if i == 1 else np.zeros(1, dtype='float32')
                             for i in df[c:c + t]['is_redacted']]
        size_of_dense_part = len(input_chat_names[0]) + len(input_sender_names[0]) \
                             + len(input_chat_type[0]) + len(input_is_redacted[0]) + len(input_dates[0])
        input_common = np.array([np.zeros(size_of_dense_part)])
        for i in range(t-1):
            input_common = np.vstack((input_common, np.hstack((input_chat_names[i], input_sender_names[i],
                                                               input_chat_type[i], input_is_redacted[i],
                                                               input_dates[i]))))

        input_common = np.delete(input_common, (0), axis=0)
        decoder_output = decoder_output_creater(decoder_input_data[c + 1:c + t], num_samples=t - 1, MAX_LEN=MAX_LEN,
                                                VOCAB_SIZE=VOCAB_SIZE)
        encoder_input = encoder_input_data[c:c + t - 1]
        decoder_input = decoder_input_data[c + 1:c + t]

        decoder_input = np.zeros(decoder_input.shape)
        say_what = model.predict([encoder_input, decoder_input, input_common])
        print(say_what)
        break

        # history = model.fit([encoder_input, decoder_input, input_common],
        #           [decoder_output, output_chat_names_tensor, output_dates_tensor], epochs = loop_in_epoch, batch_size=batch_size, verbose=0)
        c += num_samples
        if c > all_seqs:
            flag = False
    print(j, 'epoch comlete', time.time() - timing)
    timing = time.time()
    # serialize weights to HDF5
    model.save_weights("drive/My Drive/Colab Notebooks/model.h5")
    print("Saved model to disk +", loop_in_epoch)


# serialize model to JSON
# with open("model.json", "w") as json_file:
#     json_file.write(model.to_json())
print("--- %s seconds ---" % (time.time() - start_time))
