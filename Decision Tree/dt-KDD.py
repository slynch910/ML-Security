import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from scipy.io import arff
from sklearn.metrics import ConfusionMatrixDisplay
import csv


##############################################
# Class holding the decision tree model for 
# the various datasets.
##############################################
class DT:
    
    ##############################################
    # Members to be used throughout the class.
    ##############################################
    
    
    ##############################################
    # KDD-99 Dataset Members
    ##############################################
    # Columns that are represented as bytes in the dataset.
    byte_cols = ["protocol_type", "service", "flag", "class"]
    
    # Dictionary to map the protcol_map categorical data to numerical data.
    protocol_index = "protocol_type"
    protocol_map = {"tcp": 0, "udp": 1, "icmp": 2}
    
    # Dictionary to map the class categorical data to numerical data.
    class_index = "class"
    class_map = {"normal": 0, "anomaly": 1}
    
    # Categories that need to have labels mapped to numerical data.
    pendingMappings = ["service", "flag"]
    
    # Number of folds for cross validation
    num_folds = 15

    # Scoring Dictionary
    scoring_dict = {
        'acc' : 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro'
    }


    ##############################################
    # Initialize the decision tree model with the 
    # corresponding dataset.    
    ##############################################
    def __init__(self, dataset_path="KDDTrain+.arff"):
        # # The KDD data is in arff format, so we need to find a way to import it
        self.dataset = pd.DataFrame(arff.loadarff(dataset_path)[0]) 
        print(self.dataset)
        
        # Preprocess the data for later use.
        self.preprocess_data()
        
        # Now, split the class labels from the actual data going to be used.
        self.X = self.dataset[self.dataset.columns[:-1]]
        self.Y = self.dataset["class"]
        
        # Split the data into training and testing data. (Using an 80-20 split)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=1)
        
    ##############################################    
    # Preprocess the data to be used in the model.
    ##############################################
    def preprocess_data(self):
        # The dataset has a few b'<string>' values so we need to change them into actual strings.
        for col in self.byte_cols:
            self.dataset[col] = self.dataset[col].str.decode("utf-8")
            
        # Next, we need to convert the protocol categorical data into numerical data.
        self.dataset = self.dataset.replace({self.protocol_index: self.protocol_map})
        
        # Do the same for the class categorical data.
        self.dataset = self.dataset.replace({self.class_index: self.class_map})

        # Now, we need to map other data to numerical data without having a long list of every possible value.
        for col in self.pendingMappings:
            mappings = {}
            # Get the unique values in the column. (set removes duplicates)
            for i, value in enumerate(list(set(self.dataset[col]))):
                # Create a mapping that maps the value to a new index.
                mappings.update({value: i})
                
            # Replace the values in the column with the new numerical values.   
            self.dataset = self.dataset.replace({col: mappings})

    ##############################################
    # Train the model using the training data.
    ##############################################
    def train(self):
        # Create the decision tree classifier
        self.classifier = DecisionTreeClassifier(criterion="entropy", max_depth=1000, max_leaf_nodes=5, max_features=50)
        
        # Train the model with the training data.
        self.classifier = self.classifier.fit(self.X_train, self.Y_train)
        
        # Predict the labels of the testing data.
        self.y_pred = self.classifier.predict(self.X_test)
        
        # Print out the metrics from this prediction.
        self.print_metrics()
        
        # Perform cross validation on the model.
        self.perform_cross_validation()

    ##############################################
    # Print out the metrics of the model being
    # predicting the testing data.
    ##############################################
    def print_metrics(self):
        print("accuracy: ", metrics.accuracy_score(self.Y_test, self.y_pred))
        print("========== Classification Report ==========")
        print(metrics.classification_report(self.Y_test, self.y_pred, digits=3))
        print("========== Confusion Matrix ==========")
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        cm = metrics.confusion_matrix(self.Y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        
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
        
        results = []
        
        # Print out the results of the cross validation for each fold. 
        print("=======================================================")
        for i in range(0,self.num_folds):
          results.append([scores["test_acc"][i], scores["test_prec_macro"][i], scores["test_rec_macro"][i]])
          print("Accuracy rating for k={}: {}".format(i+1, scores["test_acc"][i]))
          print("Precision rating for k={}: {}".format(i+1, scores["test_prec_macro"][i]))
          print("Recall rating for k={}: {}".format(i+1, scores["test_rec_macro"][i]))
          print("=======================================================")
        
        with open("dt_results.csv", "w") as out:
            csv_out = csv.writer(out)
            csv_out.writerow(["Accuracy", "Precision", "Recall"])
            csv_out.writerows(results)


if __name__ == "__main__":
    dt = DT()
    dt.train()
    


