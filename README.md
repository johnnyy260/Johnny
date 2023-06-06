

# Counterfactual Explanations for Neural Recommenders
This repository contains data and the implementation of the modified ACCENT framework for the neural recommender: Neural Collaborative Filtering (NCF).


## Environment
To use this code, the following python version is required:
- Python 3.7.16


## Dataset
We use the popular MovieLens 100K dataset (https://grouplens.org/datasets/movielens/100k/), which contains 100K ratings on a 1 − 5 scale by 943 users on 1682 movies.


### NCF
For NCF, the data is in ```NCF/data/movielens_train.tsv```. 
Each row consists of 4 tab-separated columns, representing an interaction between a user and a movie. The columns are:
- User ID
- Item ID
- Rating (integer, from 1 to 5)
- Timestamp


## Training Models
From the unzipped folder, run the following file to start training an NCF model.
`train.py`
In the `get_rec.py` file in the `get_rec` function, you can specify the number of recommendations the model has to predict for each user

## Running Experiment
For the modified Accent algorithm, run the `experiment.py`. The script will generate counterfactual explanations for each user and save it as a CSV file.
In the `main` function of the `experiment.py` file, you can specify the list of k values in the topk to be used in the experiment.
In the `accent.py` file at the `di = ` line, you can specify the number of items dropping out of the first 5 recommended items (1-5).

### Note that after the `experiment.py` is done running, rename the accent CSV file before rerunning it on a different `di` value. 
For example, when you have ran the file on `di` = 5, rename the following files:
- accent_20 to accent_20_5
- accent_50 to accent_50_5
- accent_100 to accent_100_5

## Visualising the results
To see which the counterfactual set, number of dropped items, topk and new_topk, run the notebook file `visualise.ipynb`

