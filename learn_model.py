# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.utils import shuffle
from model_construction import LSTM_architecture

### Считывание данных
n = ['text']
data_positive_current = pd.read_csv('combined_output_positive.csv', sep=';', names=n, usecols=['text'])
total_rows = len(data_positive_current)
sample_size = total_rows // 2
reduced_data = data_positive_current.sample(n=sample_size, random_state=42)  # random_state для воспроизводимости
reduced_data.to_csv('combined_output_positive.csv', sep=';', index=False, header=False)

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
from sklearn.utils import shuffle

reviews, labels = shuffle(reviews_withoutshuffle, labels_withoutshuffle, random_state=0)


### Токенизация

def tokenize():
    punctuation_to_remove = '"#$%&\'()*+-/:;<=>[\]^_`{|}~'
    all_reviews = 'separator'.join(reviews)
    all_reviews = all_reviews.lower()
    all_text = ''.join([c for c in all_reviews if c not in punctuation_to_remove])
    texts_split = all_text.split('separator')
    all_text = ' '.join(texts_split)
    words = all_text.split()
    return words, texts_split


words, texts_split = tokenize()

new_reviews = []
for review in texts_split:
    review = review.split()
    new_text = []
    for word in review:
        if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
            new_text.append(word)
    new_reviews.append(new_text)

counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii - 1 for ii, word in enumerate(vocab, 1)}
reviews_ints = []
for review in new_reviews:
    reviews_ints.append([vocab_to_int[word] for word in review])


def add_pads(reviews_ints, seq_length):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        if len(row) == 0:
            continue
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features

### Разделение на обучающую, валидационную и тестовую выборки

features = add_pads(reviews_ints, seq_length=20)
split_frac = 0.8 # 80% на обучающую выборку

split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = labels[:split_idx], labels[split_idx:]
test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

### Определение режима: GPU или CPU

train_gpu=torch.cuda.is_available()

###Выбор гиперпараметров и инициализация сети


vocab_size = len(vocab_to_int)+1
output_size = 1
embedding_dim = 200
hidden_dim = 128
n_layers = 2
model = LSTM_architecture(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

### Обучение модели

epochs = 10 #оптимальное количество эпох для того, чтобы модель достаточно обучилась, но не переобучилась
counter = 0
batch_num = 100
clip = 5
if (train_gpu):
    model.cuda()
num_correct = 0
model.train()

best_val_loss = float('inf')
patience = 4
patience_counter = 0

for e in range(epochs):
    h = model.init_hidden_state(batch_size)
    for inputs, labels in train_loader:
        num_correct = 0
        counter += 1
        if(train_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()
        h = tuple([each.data for each in h])
        model.zero_grad()
        output, h = model.forward(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if counter % batch_num == 0:
            val_h = model.init_hidden_state(batch_size)
            val_losses = []
            model.eval()
            for inputs, labels in valid_loader:
                val_h = tuple([each.data for each in val_h])
                if(train_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()
                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())

                #accuracy
                pred = torch.round(output.squeeze())
                correct_tensor = pred.eq(labels.float().view_as(pred))
                correct = np.squeeze(correct_tensor.numpy()) if not train_gpu else np.squeeze(correct_tensor.cpu().numpy())
                num_correct += np.sum(correct)
                valid_acc = num_correct/len(valid_loader.dataset)

            model.train()
            current_val_loss = np.mean(val_losses)
            print("Epoch: {} ;".format(e + 1),
                  "Batch Number: {};".format(counter),
                  "Train Loss: {:.4f} ;".format(loss.item()),
                  "Valid Loss: {:.4f} ;".format(current_val_loss),
                  "Valid Accuracy: {:.4f}".format(valid_acc))
            # Ранняя остановка
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                #torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Ранняя остановка: нет улучшений в валидационных потерях")
                    break

        if patience_counter >= patience:
            break

PATH = "model_check_last.pt"
torch.save(model.state_dict(), PATH)