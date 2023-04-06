import numpy as np, sklearn
from scipy.io import arff
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, tree   
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

class RandomForest:
    def __init__(self, dataset_path="NSL-KDD/KDDTrain+.arff"):
        """ Initialize the Random Forest model. """

        # Load model into DataFrame from ARFF file.
        self.dataset = pd.DataFrame(arff.loadarff(dataset_path)[0])
        
        # Clean the data and fix mappings 
        self.clean_data() 

        # Split the data into training and testing sets.
        self.X, self.Y = self.dataset.loc[:, self.dataset.columns != "class"], self.dataset["class"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=0)

    def clean_data(self): 
        for cols in ["protocol_type", "service", "flag", "class"]: 
            self.dataset[cols] = self.dataset[cols].str.decode("utf-8")

        self.dataset = self.dataset.replace({"protocol_type": {"tcp": 0, "udp": 1, "icmp": 2,}})
        self.dataset = self.dataset.replace({"class": {"normal": 0, "anomaly": 1}})

        categories_that_need_mappings = ["service", "flag"]
        for category in categories_that_need_mappings: 
            mappings = {}
            for i, value in enumerate(list(set(self.dataset[category]))):
                mappings.update({value: i}) 

            self.dataset = self.dataset.replace({category: mappings})

    def train(self):
        """ Train the Random Forest model. """
        # creating a RF classifier
        self.model = RandomForestClassifier(n_estimators = 100)  
        
        # Training the model on the training dataset fit function is used to train the model using the training sets as parameters
        self.model.fit(self.x_train, self.y_train)
        
        # performing predictions on the test dataset
        self.y_pred = self.model.predict(self.x_test)

    def metrics(self):
        """ Evaluate the Random Forest model. """
        # print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(self.y_test, self.y_pred))

        # Predicting on the test data.
        print("Accuracy: ", metrics.accuracy_score(self.y_test, self.y_pred))
        print("========= Classification Report =========")
        print(metrics.classification_report(self.y_test, self.y_pred, digits=3))
        print("========= Confusion Mnatrix =========")
        print(metrics.confusion_matrix(self.y_test, self.y_pred))
        # print("========= Tree Structure =========")
        # print(tree.export_text(self.model))

    def cross_validate(self): 
        """ Cross validate the Random Forest model. """
        # result = cross_validate(self.model, self.X, self.Y, cv=3)
        # print(result)
        
        # cross_validate also allows to specify metrics which you want to see
        scores = cross_validate(self.model, self.X, self.Y, cv=10, return_train_score=True)["test_score"]
        print(scores)
        for i, score in enumerate(scores):
            print(f"Accuracy for the fold no. {i} on the test set: {score}")

if __name__ == "__main__": 
    rf = RandomForest()

    rf.train()
    rf.metrics()
    rf.cross_validate()
