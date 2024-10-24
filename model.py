import csv
import random

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

with open("data2.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence" : [float(cell) for cell in row[2:]],
            "label" : "Cancer-detected" if row[1] == "M" else "No-cancer"
        })

    holdout = int(0.50 * len(data))
    random.shuffle(data)
    testing = data[:holdout]
    training = data[holdout:]
    x_training = [row["evidence"] for row in training]
    y_training = [row["label"] for row in training]
    model.fit(x_training, y_training)

    x_testing = [row["evidence"] for row in testing]
    y_testing = [row["label"] for row in testing]
    predictions = model.predict(x_testing)

    correct = 0
    incorrect = 0
    total = 0
    for actual, predicted in zip(y_testing, predictions):
        total += 1
        if actual == predicted:
            correct += 1
        else:
            incorrect += 1

    print(f"Result for model {type(model).__name__}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Acurracy: {100 *correct / total :.2f}")