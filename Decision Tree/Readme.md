# Decision Tree

The decision tree is a classification algorithm that works by asking questions that apply to the 
diffferent features of the data in a dataset. The tree then splits based on the answers that the model deems
valuable.

The way that a split occurs in my model is through the entropy that a decision provides. Entropy is way
to quantify uncertainty or randomness. If a decision has a high level of entropy, then there's a lot of randomness
that was gotten from this. However, if there is a low level of entropy, then there's not much randomness.

In other terms, if there is a high entropy rating, then there is a low information gain. And if there is a low
entropy rating, then you get a lot more information as the data is more uniform in regards to **that** feature or attribute.


## Tuning Hyperparameters

_For the Kitsune Database:_ For this database, I had tried multiple configurations with the parameters for the
decision tree model. Due to the nature of the dataset and its numerous attributes, the depth of the tree can be very high to
accomodate this. The model that had worked the best through normal training, predicitions and cross-validation methods
included having the "entropy" criterion, a max_depth of 1000, 5 max leaf nodes and then having a max of 50 features to 
calculate the information gain from the entropy calculations.



## Cross-Validation

_For the Kitsune Database:_ For this database, I had simply had the cross validation function perform according
to the K-Fold algorithm and had set K to be 15.
