# Galaxy Classification

This repository contains code for classifying galaxies using unsupervised machine learning algorithms, specifically decision trees and random forests, applied to the Galaxy Zoo dataset. The Galaxy Zoo dataset provides various features and classifications of galaxies contributed by users, which are utilized for training and evaluating the classifiers.

## Requirements

To run the scripts in this repository, ensure you have the following dependencies installed:

- Python (version >= 3.6)
- Required Python packages:
  - numpy~=1.26.4
  - matplotlib~=3.8.2
  - statistics~=1.0.3.5
  - scikit-learn~=1.4.0
  - tabulate~=0.9.0

You can install the required Python packages using pip.


## Files

1. **decision_tree_classifier.py**: This Python script performs galaxy classification using a decision tree classifier. It uses the features provided in the Galaxy Zoo dataset to train the classifier. The classification results are evaluated using 10-fold cross-validation.

2. **random_forest_classifier.py**: This Python script implements galaxy classification using a random forest classifier with 50 trees. Similar to the decision tree classifier, it utilizes the features from the Galaxy Zoo dataset for training and evaluates the classification performance using 10-fold cross-validation.

3. **accuracy_comparison.py**: This Python script compares the accuracy of the decision tree classifier and the random forest classifier on the same dataset after 10 iterations. It computes and displays the tabulated results, allowing for a direct comparison of the performance of both classifiers.

4. **galaxy_catalogue.npy**: This file contains the dataset in numpy array format. It includes all the features and classifications required for galaxy classification.

5. **results.txt**: This file contains the results of the accuracy comparison between the decision tree classifier and the random forest classifier. The results are tabulated and can be used for evaluating the performance of the classifiers.

## Usage

To use the provided scripts for galaxy classification:

1. Ensure you have Python installed on your system.
2. Clone this repository to your local machine.
3. Install the required dependencies listed in `requirements.txt`.
4. Run `decision_tree_classifier.py` and `random_forest_classifier.py` to train and evaluate the respective classifiers.
5. Finally, run `accuracy_comparison.py` to compare the accuracy of both classifiers.

## Dataset

The Galaxy Zoo dataset used for galaxy classification contains the following columns:
- **Colors**: 
    - u-g, g-r, r-i, i-z
- **Eccentricity**:
    - ecc
- **4th Adaptive Moments**:
    - m4_u, m4_g, m4_r, m4_i, m4_z
- **50% Petrosian**:
    - petroR50_u, petroR50_r, petroR50_z
- **90% Petrosian**:
    - petroR90_u, petroR90_r, petroR90_z

## Results

The results of the accuracy comparison between the decision tree classifier and the random forest classifier are available in the tabulated format in the repository.
