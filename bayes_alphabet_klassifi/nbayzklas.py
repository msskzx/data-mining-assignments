import os
import pandas as pd
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import collections

# Gaussian Naïve Bayes Classifier

class GausNB_Klassifier:

    # initialize the classifier
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.len = len(train)
        self.prior()
        self.mean_variance()

    # map each class count to its class
    def prior(self):
        counts = self.train["Class"].value_counts().to_dict()
        self.priors = {(key, value / self.len) for key, value in counts.items()}

    # calculate the probability of x given its mean and variance
    @staticmethod
    def probability(x, mean, variance):
        if variance == 0:
            return 0.1
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * variance)))
        p = 1 / (math.sqrt(2 * math.pi * variance)) * exponent
        if p <0.1:
            return 0.1
        return p

    def mean_variance(self):
        self.mean_variance = {}
        for clss in self.train["Class"].unique():
            filtered = self.train[(self.train['Class'] == clss)]
            m_v = {}

            attr_names = list(self.train.columns.values)
            attr_names.remove("Class")

            for attr_name in attr_names:
                m_v[attr_name] = []
                m_v[attr_name].append(filtered[attr_name].mean())
                m_v[attr_name].append(math.pow(filtered[attr_name].std(), 2))

            self.mean_variance[clss] = m_v

    def predict(self):
        predictions = {}
        row_num = 0
        i = 0
        for _, row in self.test.iterrows():
            results = {}
            for key, value in self.priors:
                p = 0

                attr_names = list(self.train.columns.values)
                attr_names.remove("Class")

                for attr_name in attr_names:
                    prob = self.probability(row[attr_name],
                        self.mean_variance[key][attr_name][0],
                        self.mean_variance[key][attr_name][1])
                    if prob > 0:
                        p += math.log(prob)

                results[key] = math.log(value) + p

            predictions[i] = max([k for k in results.keys() if
                results[k] == results[max(results, key=results.get)]])
            i += 1

        return predictions

def accuracy(test, predictions):
    correct = 0
    row_num = 0
    correct_counts = {}

    for _, t in test.iterrows():
        if t["Class"] == predictions[row_num]:
            correct += 1
            if t["Class"] in correct_counts.keys():
                correct_counts[t["Class"]] += 1
            else:
                correct_counts[t["Class"]] = 1

        row_num += 1

    od = collections.OrderedDict(sorted(correct_counts.items()))

    return [(correct / len(test)) * 100.0, list(od.values())]

def plot_count_class(classes, correct_counts):
    plt.plot(classes, correct_counts)
    plt.ylabel('Count')
    plt.xlabel('Classes')
    plt.show()

# count represents the number of samples for each class
def parse_data(path, count):
    classes = []
    for letter in range(97, 123):
        for i in range(0, count):
            classes.append(chr(letter))

    images_names = list(os.listdir(path))
    images_names.sort()
    data = []

    for image_name in images_names:
        # image
        row = cv2.imread(path + image_name, 0).flatten()/255
        data.append(row)

    df = pd.DataFrame(data)
    df["Class"] = classes
    return df

def main():
    # parsing training data, 7 examples for each class
    train = parse_data("Train/", 7)
    print()

    # parsing test data
    test = parse_data("Test/", 2)

    # build classifier
    klassifier = GausNB_Klassifier(train, test)

    # predict
    predictions = klassifier.predict()

    #accuracy
    acc = accuracy(test, predictions)
    print(acc[0])
    plot_count_class(train["Class"].unique(), acc[1])

if __name__ == "__main__":
    main()
