import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Scoring Dictionary
scoring_dict = {
    'acc' : 'accuracy',
    'prec_macro': 'precision_macro',
    'rec_macro': 'recall_macro'
}

onlineFilePath = "datasets/mirai_combined.csv"

# The preprocessed data had added a row for the headers.
kitsune = pd.read_csv(onlineFilePath, header=0)

# Ignore the first column as this is the index, and then ignore the answer column.
feature_cols = kitsune.columns[1:-1]

# Create the feature and verdict arrays.
X = kitsune[feature_cols]
Y = kitsune["Verdict"]

# Generate a train and test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Create the decision tree.
classifier = DecisionTreeClassifier(criterion="entropy", max_depth=60, max_leaf_nodes=7, max_features=7)
classifier = classifier.fit(X_train, Y_train)

# Predicting on the test data.
y_pred = classifier.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(Y_test, y_pred))
print("========= Classification Report =========")
print(metrics.classification_report(Y_test, y_pred, digits=3))
print("========= Confusion Matrix =========")
print(metrics.confusion_matrix(Y_test, y_pred))
print("========= Tree Structure =========")
print(tree.export_text(classifier))


num_folds = 15


# Create the Kfold object for cross validation
kf = KFold(n_splits=num_folds)
        
# Perform the cross validation
scores = cross_validate(classifier, X, Y, scoring=scoring_dict, cv=kf, return_train_score=True)

# Have everything stored in an nparray for outputting to csv
results = []

# Print out the results of the cross validation for each fold. 
print("=======================================================")
for i in range(0, num_folds):
    results.append((scores["test_acc"][i], scores["test_prec_macro"][i], scores["test_rec_macro"][i]))            
    print("Accuracy rating for k={}: {}".format(i+1, scores["test_acc"][i]))
    print("Precision rating for k={}: {}".format(i+1, scores["test_prec_macro"][i]))
    print("Recall rating for k={}: {}".format(i+1, scores["test_rec_macro"][i]))
    print("=======================================================")
            
        
with open("results/dt_results_kitsune.csv", "w") as out:
    csv_out = csv.writer(out)
    csv_out.writerow(["Accuracy", "Precision", "Recall"])
    csv_out.writerows(results)



print(f"[+] Creating Confusion Matrix for Kitsune dataset...")
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(Y_test, y_pred), 
    display_labels=Y.unique()
).plot()

plt.savefig(f"results/Kitsune_confusion_matrix.png") 