import re
import json
import nltk
import contractions
import warnings
import pandas as pd
from num2words import num2words
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')
# nltk.download('all')
AVERAGE = "macro"
stopWords = set(stopwords.words('english'))
lematizer = WordNetLemmatizer()


def lowerPreprocess(text):
    return contractions.fix(text.lower())


def numberPreprocess(text):
    return re.compile(r'[\d]+').sub(lambda ch: num2words(ch.group()), text)


def remSWPreprocess(text):
    text = text.split()
    return " ".join([word for word in text if word not in stopWords])


def wordTokenizer(text):
    return nltk.word_tokenize(text)


def lemmaTokenizer(text):
    tokenizer = text.split()
    return [lematizer.lemmatize(word.lower()) for word in tokenizer]


def getMetrics(testLb, predLb, average):
    return metrics.accuracy_score(testLb, predLb), metrics.precision_score(testLb, predLb, average=average), \
           metrics.recall_score(testLb, predLb, average=average), metrics.f1_score(testLb, predLb, average=average)


def readData():
    textsData, labelsData = [], []
    with open("News_Category_Dataset_v2.json") as f:
        line = f.readline()
        while line:
            line = json.loads(line)
            if line['category'] in ['ENTERTAINMENT', 'WORLD NEWS', 'COMEDY', 'BUSINESS', 'WEIRD NEWS']:
                textsData.append(line['headline'])
                labelsData.append(line['category'])
            line = f.readline()
    return textsData, labelsData


if __name__ == '__main__':
    texts, labels = readData()  # 1

    # 2
    models = [CountVectorizer(preprocessor=lowerPreprocess, tokenizer=lemmaTokenizer, token_pattern=None,
                              max_features=1300),
              CountVectorizer(preprocessor=remSWPreprocess, tokenizer=wordTokenizer, token_pattern=None,
                              max_features=1000),
              CountVectorizer(preprocessor=numberPreprocess, tokenizer=lemmaTokenizer, token_pattern=None,
                              max_features=700)]

    bestMl = tuple(0 for _ in range(8))
    nrTrees, max_depth = 30, 30
    for i in range(len(models)):
        models[i].fit(texts)
        ftData = models[i].transform(texts).toarray()
        scalerData = preprocessing.StandardScaler()
        scalerData.fit(ftData)
        scaledData = scalerData.transform(ftData)

        # 3
        train, trainLabels = scaledData[:int(0.8 * len(scaledData))], labels[:int(0.8 * len(labels))]
        test, testLabels = scaledData[int(0.8 * len(scaledData)):], labels[int(0.8 * len(labels)):]
        rfc = RandomForestClassifier(n_estimators=nrTrees, max_depth=max_depth)
        rfc.fit(train, trainLabels)

        importance = zip(sorted(list(models[i].vocabulary_.keys())), rfc.feature_importances_)
        ft_names_score = pd.DataFrame(importance, columns=["feature", "score"]).sort_values(['score'],
                                                                                            ascending=[False])
        # 4
        rfc_pred = rfc.predict(test)
        accuracy, precision, recall, f1 = getMetrics(testLabels, rfc_pred, AVERAGE)
        print(f"\n{'Accuracy': ^20} {'Precision': ^35} {'Recall': ^25} {'F1': ^30}")
        print(f"{accuracy: ^20} {precision: ^35} {recall: ^25} {f1: ^30}")
        if accuracy > bestMl[1]:
            bestMl = (i, accuracy, ft_names_score.head(10), rfc_pred, train, trainLabels, test, testLabels)
        nrTrees += 10
        max_depth *= 5

    # 5
    print("\nTop 10 features:")
    print(f"{'Feature': ^15} {'Score': ^30}")
    index, accuracyRFC, ft_impt, rfc_pred, trainDT, trainLB, testDT, testLB = bestMl
    for _, row in ft_impt.iterrows():
        print(f"{row['feature']: ^15} {row['score']: ^30}")

    # 6
    knn = KNeighborsClassifier(n_neighbors=41, metric='manhattan')
    knn.fit(trainDT, trainLB)
    knn_pred = knn.predict(testDT)
    print("\nKNN Classifier")
    accuracyKNN, precision, recall, f1 = getMetrics(testLB, knn_pred, AVERAGE)
    print(f"\n{'Accuracy': ^20} {'Precision': ^35} {'Recall': ^25} {'F1': ^30}")
    print(f"{accuracyKNN: ^20} {precision: ^35} {recall: ^25} {f1: ^30}")

    svc = SVC(kernel='linear', C=0.001)
    svc.fit(trainDT, trainLB)
    svc_pred = svc.predict(testDT)
    print("\nSVC Classifier")
    accuracySVC, precision, recall, f1 = getMetrics(testLB, svc_pred, AVERAGE)
    print(f"\n{'Accuracy': ^20} {'Precision': ^35} {'Recall': ^25} {'F1': ^30}")
    print(f"{accuracySVC: ^20} {precision: ^35} {recall: ^25} {f1: ^30}")

    # 7
    print()
    maxAccuracy = max(accuracyRFC, max(accuracyKNN, accuracySVC))
    if maxAccuracy == accuracyRFC:
        print("Algoritm: Random Forest")
        print(metrics.classification_report(testLB, rfc_pred))
        print(metrics.confusion_matrix(testLB, rfc_pred))
    elif maxAccuracy == accuracyKNN:
        print("Algoritm: KNN")
        print(metrics.classification_report(testLB, knn_pred))
        print(metrics.confusion_matrix(testLB, knn_pred))
    else:
        print("Algoritm: SVC")
        print(metrics.classification_report(testLB, svc_pred))
        print(metrics.confusion_matrix(testLB, svc_pred))
