from collections import Counter
import pandas as pd
import numpy as np
from math import ceil
import torch
from sklearn.utils import shuffle
from .model_construction import LSTM_architecture


### Считывание данных

### Считывание данных
n = ['text']
data_positive_current = pd.read_csv('combined_output_positive.csv', sep=';', names=n, usecols=['text'])
data_positive_new = pd.read_csv('new_positive_reviews.csv', sep=';', names=n, usecols=['text'])
data_positive = pd.concat([data_positive_current, data_positive_new], ignore_index=True)

data_negative_current = pd.read_csv('combined_output_negative.csv', sep=';', names=n, usecols=['text'])
data_negative_new = pd.read_csv('new_negative_reviews.csv', sep=';', names=n, usecols=['text'])
data_negative = pd.concat([data_negative_current, data_negative_new], ignore_index=True)

### Формирование сбалансированного датасета

sample_size = 40000
reviews_withoutshuffle = np.concatenate((data_positive['text'].values[:sample_size],
                                         data_negative['text'].values[:sample_size]), axis=0)
labels_withoutshuffle = np.asarray([1] * sample_size + [0] * sample_size)

assert len(reviews_withoutshuffle) == len(labels_withoutshuffle)

texts, labels = shuffle(reviews_withoutshuffle, labels_withoutshuffle, random_state=0)

def tokenize():
    punctuation = '"#$%&\'()*+,-/:;<=>?[\]^_`{|}~'
    all_reviews = 'separator'.join(texts)
    all_reviews = all_reviews.lower()
    all_text = ''.join([c for c in all_reviews if c not in punctuation])
    texts_split = all_text.split('separator')
    all_text = ' '.join(texts_split)
    words = all_text.split()
    return words, texts_split

def get_vocabulary():
    words, _ = tokenize()  # Нам нужны только words для словаря
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return vocab, vocab_to_int


def tokenize_text(test_text):
    punctuation = '"#$%&\'()*+,-/:;<=>?[\]^_`{|}~'
    test_text = test_text.lower()
    test_text = ''.join([c for c in test_text if c not in punctuation])
    test_words = test_text.split()
    new_text = []
    for word in test_words:
        if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
            new_text.append(word)
    test_ints = []
    _, vocab_to_int = get_vocabulary()
    mas_to_int = []
    for word in new_text:
        if word in vocab_to_int:
            mas_to_int.append(vocab_to_int[word])
    test_ints.append(mas_to_int)

    return test_ints


def add_pads(texts_ints, seq_length):
    features = np.zeros((len(texts_ints), seq_length), dtype=int)

    for i, row in enumerate(texts_ints):
        if len(row) == 0:
            continue
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features

train_gpu=torch.cuda.is_available()

def predict(model, test_review, sequence_length=30):
    model.eval()
    test_ints = tokenize_text(test_review)
    seq_length=sequence_length
    features = add_pads(test_ints, seq_length)
    feature_tensor = torch.from_numpy(features)
    batch_size = feature_tensor.size(0)
    h = model.init_hidden_state(batch_size)
    if(train_gpu):
        feature_tensor = feature_tensor.cuda()
    output, h = model(feature_tensor, h)

    pred = torch.round(output.squeeze())
    probability = output.item()
    type_of_tonal = "Позитивное сообщение" if pred.item() == 1 else "Негативное сообщение"
    print('Вероятность положительного ответа {:.6f}'.format(output.item()))

    return type_of_tonal, probability

def scan(string):

    #file_name = 'test.txt'
    #f = open(file_name, 'r')
    #data = f.readlines()
    #input_text = ''.join(data)
    input_text = string #'Лучше бы играл Киану Ривз. Сюжет могли бы сделать поинтереснее. И наверняка стоило сменить режиссера на более одаренного'
    #f.close()

    test_ints = []
    _, vocab_to_int = get_vocabulary()

    vocab_size = len(vocab_to_int)+1
    output_size = 1
    embedding_dim = 200
    hidden_dim = 128
    number_of_layers = 2
    model = LSTM_architecture(vocab_size, output_size, embedding_dim, hidden_dim, number_of_layers)
    model.load_state_dict(torch.load("model_check.pt", weights_only=True))
    model.eval()
    seq_length = 30

    type_of_tonal, pos_prob = predict(model, input_text, seq_length)

    print("Окраска - {}, вероятность = {}%".format(type_of_tonal, 0))

    if type_of_tonal == "Негативное сообщение":
        prob = ceil((1 - pos_prob) * 100)
        return "Не рекомендую"
    else:
        prob = ceil(pos_prob * 100)
        return "Рекомендую"

if __name__ == "__main__":
    scan("вряд ли пойду пересматривать, скучно и однообразно. для чего убили главного героя в конце?")