import pandas as pd
import tensorflow.compat.v1 as tf
from accent import Accent
from helper import get_model
from helper2 import init_all_results, append_result

def generate_cf(algo, ks):
    """
    Generate counterfactual explanations for multiple k values using the specified algorithm.
    
    Args:
        algo: The algorithm used to generate explanations.
    	ks: List of k values to consider.

    Returns:
        None. The function writes the results to csv files.
    """
    # Initialize Accent as the algorithm for explanation
    explaner = Accent()

    # Ensure the values in ks are in ascending order, and the last one is 100
    for i in range(len(ks) - 1):
        assert ks[i] < ks[i + 1]
    assert ks[-1] == 100

    # Initialize the results for all k values
    all_results = init_all_results(ks)

    # Get the recommender model
    model = get_model(use_recs=True)

    # Iterate over the users to generate explanations for each
    for user_id in range(300):
        print('testing user', user_id)

        # Reset the default graph to avoid the accumulation of tensor nodes
        tf.reset_default_graph()

        # Get the recommender model again, in case it was modified during the last iteration
        model = get_model(use_recs=True)

        # Generate the counterfactual explanations for the current user
        res = explaner.find_counterfactual_multiple_k(user_id, ks, model, None, None)

        # Append the results to all_results
        append_result(ks, all_results, user_id, res)

    # Write the results to csv files, one file for each k value
    for j in range(len(ks)):
        df = pd.DataFrame(all_results[j])
        df.to_csv(f'{algo}_{ks[j]}.csv', index=False)

if __name__ == "__main__":
    # Generate counterfactual explanations using the accent algorithm
    generate_cf(algo='accent', ks=[20, 50, 100])