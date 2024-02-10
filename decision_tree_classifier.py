import numpy as np
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from support_functions import calculate_accuracy, generate_features_targets, plot_confusion_matrix
import matplotlib.pyplot as plt


# splitdata_train_test function here
def splitdata_train_test(data, fraction_training):
    np.random.seed(0)
    np.random.shuffle(data)
    split = int(len(data) * fraction_training)
    return data[:split], data[split:]


def dtc_predict_actual(data):
    # split the data into training and testing sets using a training fraction of 0.7
    train, test = splitdata_train_test(data, 0.7)

    # generate the feature and targets for the training and test sets
    # i.e. train_features, train_targets, test_features, test_targets
    train_features, train_targets = generate_features_targets(train)
    test_features, test_targets = generate_features_targets(test)

    # instantiate a decision tree classifier
    dtc = DecisionTreeClassifier()

    # train the classifier with the train_features and train_targets
    dtc.fit(train_features, train_targets)

    # get predictions for the test_features
    predictions = dtc.predict(test_features)

    # return the predictions and the test_targets
    return predictions, test_targets


if __name__ == '__main__':
    data = np.load('galaxy_catalogue.npy')

    # split the data
    features, targets = generate_features_targets(data)

    # train the model to get predicted and actual classes
    dtc = DecisionTreeClassifier()
    predicted = cross_val_predict(dtc, features, targets, cv=10)

    # calculate the model score using your function
    model_score = calculate_accuracy(predicted, targets)
    print("Our accuracy score:", model_score)

    # calculate the models confusion matrix using sklearns confusion_matrix function
    class_labels = list(set(targets))
    model_cm = confusion_matrix(y_true=targets, y_pred=predicted, labels=class_labels)

    # Plot the confusion matrix using the provided functions.
    plt.figure()
    plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
    plt.show()
