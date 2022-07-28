import torch
import sklearn.datasets as skd
import torch.nn as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
NUMBER_OF_CLASSES, BATCH_SIZE, NR_EPOCHS = 3, 70, 20


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, k):
        return self.data["arrays"][k], self.data["labels"][k]

    def __len__(self):
        return len(self.data["labels"])


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequencial = nn.Sequential(
            torch.nn.Linear(200, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, NUMBER_OF_CLASSES))

    def forward(self, seq):
        return self.sequencial(seq)


class DropOutModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = 0.4
        self.sequencial = nn.Sequential(
            torch.nn.Linear(200, 150),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.p),
            torch.nn.Linear(150, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.p),
            torch.nn.Linear(100, 50),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=self.p),
            torch.nn.Linear(50, NUMBER_OF_CLASSES))

    def forward(self, seq):
        return self.sequencial(seq)


def createDataloader(infoArr, infoLB):
    return torch.utils.data.DataLoader(MyDataset({
        "arrays": infoArr,
        "labels": infoLB
    }), batch_size=BATCH_SIZE)


def trainModel(model, loader, loss_fn):
    model.train()
    trainLoss, trainTrueLB, trainPredLB = 0, [], []

    for (arrays, labels) in loader:
        optimizer.zero_grad()
        output = model(arrays.float())
        loss = loss_fn(output, labels.long())
        trainLoss += loss.item()
        trainTrueLB.extend(labels.tolist())
        trainPredLB.extend(output.max(1)[1].tolist())
        loss.backward()
        optimizer.step()

    return trainLoss, trainTrueLB, trainPredLB


def evalModel(model, loader, loss_fn):
    model.eval()
    valloss, loadTrueLB, loadPredLB = 0, [], []

    for (arrays, labels) in loader:
        output = model(arrays.float())
        loss = loss_fn(output, labels.long())
        valloss += loss.item()
        loadTrueLB.extend(labels.tolist())
        loadPredLB.extend(output.max(1)[1].tolist())

    return valloss, loadTrueLB, loadPredLB


def plotting(title, oY1, oY2, legendLoc):
    epochsRange = [i for i in range(1, NR_EPOCHS + 1)]
    plt.title(title)
    plt.plot(epochsRange, oY1, "-b", label="train")
    plt.plot(epochsRange, oY2, "-r", label="validation")
    plt.legend(loc=legendLoc)
    plt.show()


if __name__ == '__main__':
    # 1
    init_dataset, labels = skd.make_classification(n_samples=10000, n_features=200, n_informative=100, n_classes=3)
    dataset = [torch.from_numpy(array) for array in init_dataset]

    # 3
    dt80, dt90 = int(0.8 * len(dataset)), int(0.9 * len(dataset))
    train, validation, test = dataset[:dt80], dataset[dt80:dt90], dataset[dt90:]
    trainLB, validationLB, testLB = labels[:dt80].tolist(), labels[dt80:dt90].tolist(), labels[dt90:].tolist()
    trainDT, validationDT, testDT = createDataloader(train, trainLB), createDataloader(validation, validationLB), createDataloader(test, testLB)

    # 4
    model = Model()
    loss_fn, optimizer = torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

    # 5
    print(f"{'Epoch': ^15}{'Train Accuracy': ^32} {'Validation Accuracy': ^25}")
    accuracyTrain, accuracyVal, bestMLLoss = 0, 0, float('inf')
    trainError, valError, trainAcc, valAcc = [], [], [], []

    for epoch in range(NR_EPOCHS):
        trainLoss, trainTrueLB, trainPredLB = trainModel(model, trainDT, loss_fn)
        valloss, valTrueLB, valPredLB = evalModel(model, validationDT, loss_fn)
        # 6
        if valloss < bestMLLoss:
            bestMLLoss = valloss
            torch.save(model.state_dict(), "minLoss.pt")

        # eroarea medie pe datele de train si validare
        trainError.append(trainLoss / len(trainDT))
        valError.append(valloss / len(validationDT))
        accuracyTrain, accuracyVal = metrics.accuracy_score(trainTrueLB, trainPredLB), metrics.accuracy_score(valTrueLB,
                                                                                                              valPredLB)
        trainAcc.append(accuracyTrain)
        valAcc.append(accuracyVal)
        print(f"{(epoch + 1): ^15} {accuracyTrain: ^30} {accuracyVal: ^25}")
    # 7
    plotting('Loss', trainError, valError, 'upper right')
    plotting('Accuracy', trainAcc, valAcc, 'lower right')
    # 8
    model.load_state_dict(torch.load("minLoss.pt"))
    testLoss, testTrueLB, testPredLB = evalModel(model, testDT, loss_fn)
    print(f"\nTest Mean Loss: {testLoss / len(testDT)}")
    print(metrics.classification_report(testTrueLB, testPredLB, zero_division=1))

    # 9
    DropOutmodel = DropOutModel()
    loss_fn, optimizer = torch.nn.CrossEntropyLoss(), torch.optim.SGD(DropOutmodel.parameters(), lr=0.03, momentum=0.9)
    DropOutmodel.train()
    bestMLLoss = float('inf')

    for epoch in range(NR_EPOCHS):
        trainLoss, trainTrueLB, trainPredLB = trainModel(DropOutmodel, trainDT, loss_fn)
        valloss, valTrueLB, valPredLB = evalModel(DropOutmodel, validationDT, loss_fn)
        if valloss < bestMLLoss:
            bestMLLoss = valloss
            torch.save(DropOutmodel.state_dict(), "minLoss.pt")

    DropOutmodel.load_state_dict(torch.load("minLoss.pt"))
    testLoss, testTrueLB, testPredLB = evalModel(DropOutmodel, testDT, loss_fn)
    print(f"\nDropout Test Mean Loss: {testLoss / len(testDT)}")
    print(metrics.classification_report(testTrueLB, testPredLB, zero_division=1))