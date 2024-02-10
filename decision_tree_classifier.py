import numpy as np
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# copy your splitdata_train_test function here
def splitdata_train_test(data, fraction_training):
    np.random.seed(0)
    np.random.shuffle(data)
    split = int(len(data) * fraction_training)
    return data[:split], data[split:]


# copy your generate_features_targets function here
def generate_features_targets(data):
    # complete the function by calculating the concentrations

    target_objects = data['class']

    target_features = np.empty(shape=(len(data), 13))
    target_features[:, 0] = data['u-g'] + 1
    target_features[:, 1] = data['g-r']
    target_features[:, 2] = data['r-i']
    target_features[:, 3] = data['i-z']
    target_features[:, 4] = data['ecc']
    target_features[:, 5] = data['m4_u']
    target_features[:, 6] = data['m4_g']
    target_features[:, 7] = data['m4_r']
    target_features[:, 8] = data['m4_i']
    target_features[:, 9] = data['m4_z']

    # fill the remaining 3 columns with concentrations in the u, r and z filters
    # concentration in u filter
    target_features[:, 10] = data['petroR50_u'] / data['petroR90_u']
    # concentration in r filter
    target_features[:, 11] = data['petroR50_r'] / data['petroR90_r']
    # concentration in z filter
    target_features[:, 12] = data['petroR50_z'] / data['petroR90_z']

    return target_features, target_objects


# complete this function by splitting the data set and training a decision tree classifier
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


def calculate_accuracy(predicted_classes, actual_classes):
    return sum(actual_classes == predicted_classes) / len(actual_classes)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')


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
