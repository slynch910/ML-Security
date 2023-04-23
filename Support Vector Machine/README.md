# Overview:
This code applies a Scikit-Learn's LinearSVC model to the Iot-23, NSL-KDD, UCI, and Kitsune datasets for
the purposes of optimization and comparison of the model's performance on each of the individual datasets

## Support Vector Machines, Linear Support Vector Machines, and sklearn.LinearSVC:
A Support Vector Machine is a supervised learning model used in this instance for the purposes
of classification. They are typically used as a binary linear classifier,
assigning each datapoint into one of two categories, represented by its position
relative to a hyperplane (decision boundary). The kernel trick can be applied for an SVMs usage
in multilabel classification problems. 
Linear SVMs are applied when the training datapoints are linearly separable. 
One will be applied here because they typically have improved performance
over non-linear SVMS in multlabel classifcation problems
The choice of model is Scikit-Learn's LinearSVC, which is an implementation
of linear support vector machine classifiers.


### Data Preprocessing: 

Iot23 dataset:
I took the original Iot23 dataset and selected about 1/4 of the datapoints
for the labels with larger proportions and all of the datapoints for those with
lesser proportions. DDos attacks were not included in the selection at all and
the 1 FileDownload entry is not included because one instance is not useful to us

Label names were changed to be a bit more understandable:
PartofAHorizontalPortScan: POHPS
Malicious-C&CFileDownload : C&CFileDownload
Malicious-C&C: C&C

Other remained the same: Okiru, Benign

In the code I dropped any columns that only contained values for an incredibly small number of datapoints
(service, duration, etc.) as well as those irrelevant to the classification (uid)


UCI dataset:
I took the original UCI dataset and selected about 15,000 entries for each label
(syn, scan, benign, udp, udpplain).


### Tuning Hyperparameters and Overfitting Detection:
hyperparameters were tuned by hand. Results are included in the hyperparameter-tuning.txt file 

my choices for detecting overfitting was manual comparison of training and validation accuracies and
Stratified K-Fold cross validation to ensure that the training and validation accuracies of the individual folds
as well as the mean training and validation accuracies remained relatively consistent




