import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import csv


from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt



####################################################
# Class to hold the DT for the IoT data
####################################################
class DT:
    
    # Number of folds for cross validation
    num_folds = 15

    # Scoring Dictionary
    scoring_dict = {
        'acc' : 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro'
    }
    
    
    ####################################################
    # Constructor
    ####################################################    
    def __init__(self, datasetPath="datasets/uci_dataset.csv"):
        
        # Read in the data.
        self.dataset = pd.read_csv(datasetPath, header=0)
        
        
        print(self.dataset.head())
        
        # Split the data into data and labels
        # Data
        self.X = self.dataset[self.dataset.columns[:-1]]
        # Labels
        self.Y = self.dataset[self.dataset.columns[-1]]
        
        # Split the data into training and testing data.
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=1)
        
    ####################################################
    # Train the model with the training data.
    ####################################################
    def train(self):
        
        # Create the decision tree classifier.
        self.classifier = DecisionTreeClassifier(criterion="entropy", max_depth=32, max_leaf_nodes=7, max_features=7)
        
        # Train the model.
        self.classifier = self.classifier.fit(self.X_train, self.Y_train)
        
        # Predict the labels for the test data.
        self.y_pred = self.classifier.predict(self.X_test)
        
        # Print out the metrics from the model.
        self.print_metrics()
        
        # Perform cross validation on the model.
        self.perform_cross_validation()
    
    
    ####################################################
    # Print out the metrics for the model.
    ####################################################
    def print_metrics(self):
        print("accuracy: ", metrics.accuracy_score(self.Y_test, self.y_pred))
        print("========== Classification Report ==========")
        print(metrics.classification_report(self.Y_test, self.y_pred, digits=3))
        print("========== Confusion Matrix ==========")
        print(metrics.confusion_matrix(self.Y_test, self.y_pred))   
        print("========== Tree Structure ==========")
        print(tree.export_text(self.classifier))
        
    ##############################################
    # Perform cross validation and print results.
    ##############################################
    def perform_cross_validation(self):
        
        # Create the Kfold object for cross validation
        kf = KFold(n_splits=self.num_folds)
        
        # Perform the cross validation
        scores = cross_validate(self.classifier, self.X, self.Y, scoring=self.scoring_dict, cv=kf, return_train_score=True)
        
        # Have everything stored in an nparray for outputting to csv
        results = []
        
        # Print out the results of the cross validation for each fold. 
        print("=======================================================")
        for i in range(0,self.num_folds):
            results.append((scores["test_acc"][i], scores["test_prec_macro"][i], scores["test_rec_macro"][i]))            
            print("Accuracy rating for k={}: {}".format(i+1, scores["test_acc"][i]))
            print("Precision rating for k={}: {}".format(i+1, scores["test_prec_macro"][i]))
            print("Recall rating for k={}: {}".format(i+1, scores["test_rec_macro"][i]))
            print("=======================================================")
            
        
        print(results)
            
        
        with open("results/dt_results_uci.csv", "w") as out:
            csv_out = csv.writer(out)
            csv_out.writerow(["Accuracy", "Precision", "Recall"])
            csv_out.writerows(results)

        print(f"[+] Creating Confusion Matrix for UCI dataset...")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(self.Y_test, self.y_pred), 
            display_labels=self.Y.unique()
        ).plot()

        plt.savefig(f"results/UCI_confusion_matrix.png") 

if __name__ == "__main__":
    dt = DT()
    dt.train()
    