# Preprocessing


_Kitsune Database_: For this database, the Mirai botnet dataset was split into two datasets; namely the actual data
and their corresponding labels (benign or malicious). I had combined these two csvs into one using the pandas library. Because the
dataset was so large, I had to also cut down the number of datapoints that I was going to feed into the model.
I had added a variable that could be dynamically modified to increase or decrease the size of the dataset. It will produce a
50-50 split (benign vs malicious) into one big dataset that when added together, will equal the value in the 
variable mentioned.

I also had to add feature labels for the csv as it hadn't contained anything. This was a simple loop that would just label each cell as
"Feature X". The original Kitsune dataset did not have any labels.