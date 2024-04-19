# Gamma vs Hadron Particle Classification

## Description:
This project focuses on classifying high-energy particle events as either gamma particles (signal) or hadron particles (background) using several machine learning classification algorithms. The dataset consists of simulated data from a ground-based atmospheric Cherenkov gamma telescope, capturing the pulses left by incoming Cherenkov photons on photomultiplier tubes. These pulses form patterns known as shower images, which are used to discriminate between gamma and hadron particles.

## Dataset Information:

The dataset includes various features extracted from the shower images, such as the major and minor axes of the ellipse, the sum of content of all pixels, asymmetry measures, and angular information.
Each event is labeled as either "1" (gamma) or "0" (hadron).

## Objective:
The objective is to train machine learning models to accurately classify events as gamma or hadron particles based on the given features.

## Algorithms Used:
Several classification algorithms were employed and compared:

K-Nearest Neighbors (KNN)
Naive Bayes
Support Vector Machines (SVM)
Logistic Regression
Evaluation Metrics:
The following evaluation metrics were used to assess model performance:

## Accuracy: 
The proportion of correctly classified instances.
- Precision: The proportion of true positive predictions among all positive predictions.
- Recall: The proportion of true positive predictions among all actual positive instances.
- F1-score: The harmonic mean of precision and recall, providing a balance between the two metrics.
## Results:
Each model's performance was assessed based on these evaluation metrics. The model with higher accuracy, precision, recall, and F1-score is considered better suited for the classification task.

## Instructions:

Clone the repository to your local machine.
Install the necessary dependencies.

```bash
pip install pandas numpy scikit-learn imblearn matplotlib
```

Run the Jupyter notebook or Python script to train and evaluate the classification models.
Explore the evaluation metrics to compare the performance of different algorithms.
Adjust hyperparameters or try different algorithms to improve classification metrics if needed.

## Conclusion:
Based on the evaluation metrics, the most suitable algorithm for this classification task can be determined, considering both overall accuracy and the trade-off between precision and recall. Here SVM provides the highest results.

## Note:
For further details on the dataset and implementation, please refer to the documentation and comments within the code files. The dataset and details of it is [available here.](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)
