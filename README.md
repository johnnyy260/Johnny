

# Counterfactual Explanations for Neural Recommenders
This repository contains data and the implementation of the modified ACCENT framework for the neural recommender: Neural Collaborative Filtering (NCF).


## Environment
To use this code, the following python version is required:
- Python 3.7.16

Use the following code in Anaconda to create the environment:
`conda env create --file env.yml`
After that, manually install the following package in the new environment:
`pip install tensorflow-intel`

Note: ignore the error after installing the tensorflow-intel package

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
From the unzipped folder, run the following file to start training an NCF model and predict the recommendations for each user.
`train.py`
In the `get_rec.py` file in the `get_rec` function, you can specify the number of recommendations the model has to predict for each user.

This `train.py` imports two functions from two different modules, get_rec from get_rec module and get_model from helper module, and defines a new function called main. The main function calls get_model function with the use_recs parameter set to False, which trains a new NCF model. Then it calls get_rec function, which generates recommendations for each user based on the trained model and saves the results in a csv file called recs.csv. 

#### In order to use the pre-trained model with the 100 predictions made for each user, unzip the `scripts` folder.

## Running Experiment
For the modified Accent algorithm, run the `experiment.py`. The script will generate counterfactual explanations for each user and save it as a CSV file.
In the `main` function of the `experiment.py` file, you can specify the list of k values in the topk to be used in the experiment.
In the `accent.py` file at the `di = ` line, you can specify the number of items dropping out of the first 5 recommended items (1-5).

### Note that after the `experiment.py` is done running, rename the accent CSV file before rerunning it on a different `di` value. 
For example, when you have ran the file on `di` = 5, rename the following files:
- accent_20 to accent_20_5
- accent_50 to accent_50_5
- accent_100 to accent_100_5

1. `experiment.py`: This is the entry point of the experiment. It defines a list of k values to be used in the experiment and then calls the function generate_cf from get_counterfactual.py to generate counterfactual explanations for each k value using the algorithm specified in the command-line arguments.

2. `get_counterfactual.py`: This script generates counterfactual explanations using the modified Accent algorithm. It initializes the Accent algorithm, checks that the k values are in ascending order, and initializes the results for all k values. It then loads a recommender model and iterates over a set of users to generate explanations for each user. It then writes the results to CSV files, one for each k value.

3. `accent.py`: This script implements the modified Accent algorithm to find counterfactual explanations. Given a user and a list of k values to consider, the find_counterfactual_multiple_k method computes counterfactual explanations and returns them. This function calculates the influence of each training set item on the test loss and then finds the best counterfactual explanation for each k value.

4. `accent_template.py`: This script implements the AccentTemplate class which inherits from the ExplanationAlgorithmTemplate class. It provides a method try_replace which, given a replacement item, attempts to swap the replacement and the recommendation.

5. `helper2.py`: This script provides helper functions used in get_counterfactual.py. The init_all_results function initializes a list of dictionaries to store the results of explanation algorithms for each k value. The append_result function appends the result of an explanation algorithm to the results dataset.

In summary, these scripts work together to run an experiment for generating counterfactual explanations using the modified Accent algorithm for a given list of k values. They then store and display the results. The generated counterfactual explanations provide a way of understanding the behavior of the recommendation algorithms by showing what changes would need to be made to the input to change the outcome (recommendation).


## Visualising the results
To see which the counterfactual set, number of dropped items, topk and new_topk, run the notebook file `visualise.ipynb`

