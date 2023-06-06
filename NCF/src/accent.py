from time import time

import numpy as np

from helper import get_scores
from accent_template import AccentTemplate


class Accent(AccentTemplate):
    @staticmethod
    def find_counterfactual_multiple_k(user, ks, model, data, args):
        """
        given a user, find an explanation for that user using ACCENT
        Args:
            user: ID of user
            ks: a list of values of k to consider
            model: the recommender model, a Tensorflow Model object

        Returns: a list explanations, each correspond to one value of k. Each explanation is a tuple consisting of:
                        - a set of items in the counterfactual explanation
                        - the originally recommended item
                        - a list of items in the original top k
                        - a list of predicted scores after the removal of the counterfactual explanation
                        - a list of items in the new top k
                        - the predicted replacement item
        """
        begin = time()  # Start the timer

        # Specify the number of items dropping out of the first 5 recommended items (1-5)
        di = 5

        # Find the indices in the training set for the user
        u_indices = np.where(model.data_sets.train.x[:, 0] == user)[0]

        # Get the items visited by the user
        visited = [int(model.data_sets.train.x[i, 1]) for i in u_indices]

        # Validate the visited items
        assert set(visited) == model.data_sets.train.visited[user]

        # Initialize the influences matrix
        influences = np.zeros((ks[-1], len(u_indices)))

        # Get the scores and topk items for the user
        scores, topk = get_scores(user, ks[-1], model)

        # Calculate the influence of each training set item on the test loss
        for i in range(ks[-1]):
            test_idx = user * ks[-1] + i
            assert int(model.data_sets.test.x[test_idx, 0]) == user
            assert int(model.data_sets.test.x[test_idx, 1]) == topk[i]
            train_idx = model.get_train_indices_of_test_case([test_idx])
            tmp, u_idx, _ = np.intersect1d(train_idx, u_indices, return_indices=True)
            assert np.all(tmp == u_indices)
            tmp = -model.get_influence_on_test_loss([test_idx], train_idx)
            influences[i] = tmp[u_idx]

        # Initialize variables for the best counterfactual explanation
        res = None
        best_repl = -1
        best_i = -1
        best_gap = 1e9

        ret = []

        # Find the best counterfactual explanation for each value of k
        for k in ks:
            for i in range(5, k):
                for j in range(5):  # Loop over the first 5 recommendations
                    tmp_res, tmp_gap = Accent.try_replace(topk[i], scores[topk[j]] - scores[topk[i]], influences[j] - influences[i])
                    if tmp_res is not None and (
                            res is None or len(tmp_res) < len(res) or (len(tmp_res) == len(res) and tmp_gap < best_gap)):
                        res, best_repl, best_i, best_gap = tmp_res, topk[i], i, tmp_gap

            # If a counterfactual explanation was found
            if res is not None:
                # Calculate the predicted scores after removing the counterfactual explanation
                predicted_scores = np.array([scores[item] for item in topk[:k]])
                for item in res:
                    predicted_scores -= influences[:k, item]
                # Get the new topk items
                score_item_pairs = list(zip(predicted_scores, topk[:k]))
                score_item_pairs.sort(reverse=True)
                new_topk = [item for _, item in score_item_pairs]
                # Get the items that were dropped from the top 5
                dropped_items = [item for item in topk[:5] if item not in new_topk[:5]]
                incomming_items = [item for item in topk[5:] if item not in new_topk[5:]]

                # Try to find additional items to drop until there are at least di dropped items
                counter = 0  # Define a counter variable outside the loop
                while len(dropped_items) < di and counter < 10:  # Add a condition to check the counter
                    for i in range(5, k):
                        for j in range(5):  # Loop over the first 5 recommendations
                            if topk[j] not in dropped_items and topk[i] not in incomming_items:  # Don't replace an item that's already been replaced
                                tmp_res, tmp_gap = Accent.try_replace(topk[i], scores[topk[j]] - scores[topk[i]], influences[j] - influences[i])
                                if tmp_res is not None:
                                    res, best_repl, best_i, best_gap = tmp_res, topk[i], i, tmp_gap
                                    # recalculate predicted_scores and new_topk
                                    predicted_scores = np.array([scores[item] for item in topk[:k]])
                                    for item in res:
                                        predicted_scores -= influences[:k, item]
                                    score_item_pairs = list(zip(predicted_scores, topk[:k]))
                                    score_item_pairs.sort(reverse=True)
                                    new_topk = [item for _, item in score_item_pairs]
                                    dropped_items = [item for item in topk[:5] if item not in new_topk[:5]]
                                    incomming_items = [item for item in topk[5:] if item not in new_topk[5:]]
                                else:
                                    break
                    counter += 1  # Increment the counter at the end of each loop iteration

                ret.append((set(visited[idx] for idx in res), topk[0], topk[:k], new_topk, list(predicted_scores), best_repl))
            else:
                ret.append((None, topk[0], topk[:k], None, None, -1))

        print('counterfactual time', time() - begin)  # Print the elapsed time

        return ret