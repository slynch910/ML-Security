import pandas as pd

ATTACK_START = 121622
NUM_ENTRIES = 50000

# How to convert the values in the dataset.
translation = {0 : "Benign", 1 : "Malicious"}

# Read in the csv files.
dataset_benign = pd.read_csv('Mirai_dataset.csv', nrows=NUM_ENTRIES)
dataset_malicious = pd.read_csv('Mirai_dataset.csv', skiprows=ATTACK_START, nrows=NUM_ENTRIES)
labels_benign = pd.read_csv('mirai_labels.csv', nrows=NUM_ENTRIES)
labels_malicious = pd.read_csv('mirai_labels.csv', nrows=NUM_ENTRIES, skiprows=ATTACK_START)

# Read the whole DATA dataset up to the point that we want it. Then just separate the two.
dataset = pd.read_csv('Mirai_dataset.csv', nrows=ATTACK_START + NUM_ENTRIES)

# Append the labels to the dataset at the end. Format will be <data>,<label - 0 or 1>
smallDataset = pd.concat((dataset[:NUM_ENTRIES], dataset[ATTACK_START:ATTACK_START + NUM_ENTRIES]), axis=0)

# Read the whole LABEL dataset up to the point that we want it. Then just separate the two.
labelDataset = pd.read_csv('mirai_labels.csv', nrows=ATTACK_START + NUM_ENTRIES)
labelDataset = labelDataset.replace({'0': translation})


# Append the labels to the dataset at the end. Format will be <data>,<label - 0 or 1>
smallLabelDataset = pd.concat((labelDataset[:NUM_ENTRIES], labelDataset[ATTACK_START:ATTACK_START + NUM_ENTRIES]), axis=0)

# Combine the two datasets via columns
smallCombo = pd.concat([smallDataset, smallLabelDataset], axis=1)

smallCombo.columns = [*smallCombo.columns[:-1], 'Verdict']

print(smallCombo)


# Output the smaller dataset to a csv file.
with open('mirai_combined.csv', 'w') as f:
    smallCombo.to_csv(f, index=False)