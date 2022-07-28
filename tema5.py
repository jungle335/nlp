import pandas as pd
import nltk
import torch
from nltk.corpus import stopwords

DIM_PAD = 800
DIM_VOC = 0


def readData(numeFisier="IMDB Dataset.csv"):
    data = pd.read_csv(numeFisier)
    texts, labels = data['review'].values.tolist(), data['sentiment'].replace(['negative', 'positive'], [0, 1]).tolist()
    txt80, txt90 = int(0.8 * len(texts)), int(0.9 * len(texts))
    return texts[:txt80], labels[:txt80], texts[txt80:txt90], labels[txt80:txt90], texts[txt90:], labels[txt90:]


def tokenizer(data):
    reviews = []
    for review in data:
        reviews.append(nltk.word_tokenize(review.lower()))
    return reviews


def get_vocab(data):
    return set([word for review in data for word in review])


def put_tokens_indexes(vocab):
    tokens_indices = dict((token_list, index) for index, token_list in enumerate(vocab, start=2))
    tokens_indices['UNK'] = 0
    tokens_indices['PAD'] = 1
    return tokens_indices


def get_vectorize(data, tokens_indices):
    vectorized, tokIndKeys = [], tokens_indices.keys()
    for review in data:
        vectorized.append(
            [tokens_indices[word] if word in tokIndKeys else tokens_indices['UNK'] for word in review])

    return vectorized


def pad(samples, max_length):
    return torch.tensor([
        sample[:max_length] + [1] * max(0, max_length - len(sample))
        for sample in samples
    ])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, k):
        return self.samples[k], self.labels[k]

    def __len__(self):
        return len(self.samples)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = 0.4
        self.embedding = torch.nn.Embedding(DIM_VOC * 2, 100, padding_idx=1)
        self.conv = torch.nn.Sequential(
            torch.nn.Dropout(p=self.p),
            torch.nn.Conv1d(in_channels=100, out_channels=128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),

            torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),

            torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        global_average = torch.nn.AvgPool1d(kernel_size=DIM_PAD // 8, stride=DIM_PAD // 8)
        self.convolutions = torch.nn.Sequential(
            self.conv, global_average
        )
        flatten = torch.nn.Flatten()
        linear = torch.nn.Linear(in_features=128, out_features=2)
        self.classifier = torch.nn.Sequential(flatten, linear)

    def forward(self, input):
        embeddings = self.embedding(input)
        embeddings = embeddings.permute(0, 2, 1)
        output = self.classifier(self.convolutions(embeddings))
        return output


if __name__ == '__main__':
    train, trainLB, validation, validationLB, test, testLB = readData()
    train, validation, test = tokenizer(train), tokenizer(validation), tokenizer(test)
    vocabTrain, vocabVal, vocabTest = get_vocab(train), get_vocab(validation), get_vocab(test)

    DIM_VOC = len(vocabTrain)
    trainVect = get_vectorize(train, put_tokens_indexes(vocabTrain))
    validationVect = get_vectorize(validation, put_tokens_indexes(vocabVal))
    testVect = get_vectorize(test, put_tokens_indexes(vocabTest))

    trainVect, validationVect, testVect = pad(trainVect, DIM_PAD), pad(validationVect, DIM_PAD), pad(testVect, DIM_PAD)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_dl = torch.utils.data.DataLoader(Dataset(trainVect, trainLB), batch_size=64, shuffle=True)
    validation_dl = torch.utils.data.DataLoader(Dataset(validationVect, validationLB), batch_size=64, shuffle=True)
    test_dl = torch.utils.data.DataLoader(Dataset(testVect, testLB), batch_size=64, shuffle=False)

    best_val_acc = 0
    for epoch_n in range(10):
        print(f"Epoch #{epoch_n + 1}")
        model.train()
        for (inputs, targets) in train_dl:
            model.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        all_predictions, all_targets = torch.tensor([]), torch.tensor([])
        for (inputs, targets) in validation_dl:
            with torch.no_grad():
                output = model(inputs)

            predictions = output.argmax(1)
            all_targets = torch.cat([all_targets, targets])
            all_predictions = torch.cat([all_predictions, predictions])

        val_acc = (all_predictions == all_targets).float().mean().numpy()
        print(f"Accuracy: {val_acc}\n")

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "./model")
            best_val_acc = val_acc

    print("\nBest validation accuracy", best_val_acc)
    model.load_state_dict(torch.load("./model"))
    model.eval()
    all_predictions, all_targets = torch.tensor([]), torch.tensor([])
    for (inputs, targets) in test_dl:
        with torch.no_grad():
            output = model(inputs)

        predictions = output.argmax(1)
        all_targets = torch.cat([all_targets, targets])
        all_predictions = torch.cat([all_predictions, predictions])