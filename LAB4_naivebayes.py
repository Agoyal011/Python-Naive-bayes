import numpy as np
import pandas as pd
from copy import copy

# returns the train and test sets with given proportions


def my_function(dataframe, proportion_tr, proportion_test):
    train = dataframe.sample(frac=proportion_tr, random_state=200)  # random state is a seed value
    test = dataframe.drop(train.index)

    return train, test


class NaiveBayes:


    def fit(self, X, y):
        y = [ele[0] for ele in y]
        n_samples, n_features = X.shape
        # n_samples => 398 | n_features => 31

        # gets the unique class labels
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        # classes => B | M -> length => 2

        # to calculate the mean,var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        print(y)
        for index, c in enumerate(self._classes):
            # X is the train values
            X_c = X[y[index] == c][0]
            print(X_c)

            self._mean[index, :] = X_c.mean(axis=0)
            self._var[index, :] = X_c.var(axis=0)
            self._priors[index] = X_c.shape[0] / float(n_samples)


        # where X is the test set

    def predict(self, X):

        # predicting the target values in the test set

        y_pred = [self._predict(x) for x in X]

        return np.array(y_pred)

    def _predict(self, x):
        # P(C|X) = (P(X|C) n P(C)) / P(X) where P(C/X) is the posterior prob
        #
        posteriors = []

        # to calculate posterior probability for each class
        for index, c in enumerate(self._classes): # loop is executed 2 times
            prior = np.log(self._priors[index])

            # using formula to calculate the posterior prob
            posterior = np.sum(np.log(self._pdf(index, x)))
            # adding it with the prior probability
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

        # probability density function ((P(X|C))

    def _pdf(self, class_idx, x):
        # calculating the mean as part of the formula
        mean = self._mean[class_idx]
        # calculating the variance as part of the formula
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


if __name__ == "__main__":
    df = pd.read_csv("C:/Users/Aniket/AI_datasets/new_bc_data.csv")

    df = df.drop("Unnamed: 0", axis=1)

    df = df.rename(columns={"Diagnosis": "label"})

    # calling the function which will split the df into 70% train and 30% test
    train, test = my_function(df, 0.7, 0.3)

    # pops the last column/label column of the dataset

    train.pop(train.columns[-1])
    # print(X_train)
    X_train = train.to_numpy()
    # print(X_train)

    train1, test1 = my_function(df, 0.7, 0.3)
    y_train_df = train1.iloc[:, -1:]
    # print(y_train_df)
    # Converting labels to numpy array
    y_train = y_train_df.to_numpy()
    y_train = copy(y_train.reshape(-1, 1))
    # print("The y_train target values are ", (y_train))

    train, test = my_function(df, 0.7, 0.3)

    # calling the function again

    test.pop(test.columns[-1])
    X_test = test.to_numpy()
    # print("The X_test values are :", X_test)

    train1, test1 = my_function(df, 0.7, 0.3)
    y_test_df = test1.iloc[:, -1:]
    # print(y_train_df)
    y_test = y_test_df.to_numpy()
    y_test = copy(y_test.reshape(-1, 1))


    # print("The Y_test target values are :", y_test)

    # function to test the accuracy

    def accuracy(y_real, y_pred):
        accuracy = np.sum(y_real == y_pred) / len(y_real)
        return (accuracy-10)

    # calling the class and other methods
    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # prints the accuracy of the classifier

    print("Naive Bayes classification accuracy is ", accuracy(y_test, predictions))
