import numpy as np
from statistics import mean, median, stdev

from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

from random_forest_classifier import rf_predict_actual
from support_functions import generate_features_targets, calculate_accuracy


def compare_accuracy(decision_tree_accuracy_scores, random_forest_accuracy_scores):

    dt_median = median(decision_tree_accuracy_scores)
    rf_median = median(random_forest_accuracy_scores)

    dt_mean = mean(decision_tree_accuracy_scores)
    rf_mean = mean(random_forest_accuracy_scores)

    dt_sd = stdev(decision_tree_accuracy_scores)
    rf_sd = stdev(random_forest_accuracy_scores)

    column1 = ['median scores', 'mean scores', 'standard deviation of scores']
    headers = ['Random Forest', 'Decision Tree']
    table_data = [column1, [rf_median, rf_mean, rf_sd], [dt_median, dt_mean, dt_sd]]

    table = tabulate(zip(*table_data), headers, tablefmt="grid", floatfmt='.3f')

    print(table)

    if dt_median < rf_median:
        percent_difference = 100 - dt_median / rf_median * 100
        print(f'Random forest classifier is {percent_difference:.2f} % more accurate than Decision tree classifier')

    else:
        percent_difference = 100 - rf_median / dt_median * 100
        print(f'Decision classifier is {percent_difference:.2f} % more accurate than Random forest classifier')


def dt_classifier(data):

    # split the data
    features, targets = generate_features_targets(data)

    # train the model to get predicted and actual classes
    dtc = DecisionTreeClassifier()
    predicted = cross_val_predict(dtc, features, targets, cv=10)

    # calculate the model score using your function
    model_score = calculate_accuracy(predicted, targets)

    return model_score


def rf_classifier(data):

    # get the predicted and actual classes
    number_estimators = 50              # Number of trees
    predicted, actual = rf_predict_actual(data, number_estimators)

    # calculate the model score using your function
    model_score = calculate_accuracy(predicted, actual)

    return model_score


def calculate_scorelist(data, iterations=10):

    rfc_scorelist = []
    dtc_scorelist = []

    for _ in range(iterations):

        rfc_score = rf_classifier(data)
        dtc_score = dt_classifier(data)

        rfc_scorelist.append(rfc_score)
        dtc_scorelist.append(dtc_score)

    return dtc_scorelist, rfc_scorelist


def results(data, iterations):

    print(f'Number of iterations: {iterations}')
    dtc_scorelist, rfc_scorelist = calculate_scorelist(data, iterations)
    compare_accuracy(dtc_scorelist, rfc_scorelist)


if __name__ == '__main__':

    data = np.load('galaxy_catalogue.npy')

    iterations = 10
    results(data, iterations)
