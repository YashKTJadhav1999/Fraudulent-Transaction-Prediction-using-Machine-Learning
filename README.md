# Fraudulent-Transaction-Prediction-using-Machine-Learning


Aim: The Machine Learning model aims to predict fraudulent transactions in a financial company.

In a Fraud detection model, the PRECISION of the model is highly important. Other factors like accuracy, f1_score, and auc_roc score are also important. Instead of predicting Legitimate transactions, our focus is to predict Fraud transactions, thus we have to be precise with our predictions.

MULTICOLINEARITY:
The dataset provided has attributes that are highly correlated to each other, we can see that by the Variance Inflation Factor. The highly correlated attributes are combined into one attribute and the individual attributes are dropped.

SELECTION OF THE MODEL:
Checking if the data is unbalanced, is a very important part of model-building as unbalanced data can give biased predictions. The dataset provided is highly unbalanced as the percentage of Legit data is 99.8709% and that of Fraud data is 0.1291%.
To resolve this unbalanced data we will use Random Forest Classifier and Decision Tree Classifier.
Models like XGBoost, Bagging, ANN, and Logistic Regression may give good accuracy but they won't give good precision and recall values.

SPLITTING THE DATASET INTO TEST AND TRAIN:
Before building and evaluating the model, we will split the data into the Training dataset and the Test dataset. But before that, we will create a copy of the dataset given and then split it into train and test. I have kept the test_size = 0.3, meaning 30% of the data will be used for the test set and the remaining 70% for the training set.

Decision Tree: Decision trees in machine learning provide an effective method for making decisions because they lay out the problem and all the possible outcomes

Random Forest: Random Forest reduces overfitting by averaging multiple decision trees and is less sensitive to noise and outliers in the data.

EVALUATION SCORE OF THE MODELS:
After we build the model, we evaluate them by removing the score.
The score of the Decision Tree Classifier: 99.92293531071581
The score of Random Forest Classifier: 99.95903155199169
The scores of both models are almost similar.

CONFUSION MATRIX:
To further evaluate the models we will use Confusion Matrix to represent the prediction summary in matrix form. The key terms of the Confusion Matrix are True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN). While comparing these key terms, we found that the True Negatives of the Random Forest are less than the True Negatives of the Decision Tree. Other key terms were almost similar. So this tells us that Random Forest is better than Decision Tree.

CLASSIFICATION REPORT:
Another evaluation technique used was the Classification Report. The Classification Report gives a detailed breakdown of how well your model performs on each class, and how it balances the trade-off between precision and recall. It returns us with the values of f1_score, precision, recall, and support. When we compare the values of the Random Forest and Decision Tree, we can see that the precision of the Random forest for predicting fraudulent transactions is 0.97 and the f1_score for predicting fraudulent transactions is 0.81. Whereas, that of the Decision Tree is 0.70 in terms of both precision and f1_score. Thus, we can conclude that a Random Forest is better than a Decision Tree.

ROC_AUC SCORE:
Another way of evaluating the models is by ROC AUC score and curve. The AUC is widely used to measure the accuracy of diagnostic tests. The closer the ROC curve is to the upper left corner of the graph, the higher the accuracy of the test because in the upper left corner, the sensitivity = 1 and the false positive rate = 0 (specificity = 1). The ideal ROC curve thus has an AUC = 1.0. The AUC(Decision Tree) is 0.8519 and AUC(Random Forest) is 0.8521. As the ROC AUC score of both the models is above 0.8, we can consider both of the models to be working pretty well.

CONCLUSION:
To conclude, we can see both the models are efficient and are providing good results. However, the Random Forest model has slightly better results than the Decision Tree model. So we will print the predicted values of the Random Forest classifer model into a CSV format.
