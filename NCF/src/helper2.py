import hashlib
import os
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd


def init_all_results(ks):
    """
    init a list of results to store explanations produced by explanation algorithms
    :param ks: list of k values to considered
    :return: a list of dictionaries where each one stores the result of one k value
    """
    all_results = []
    for _ in ks:
        all_results.append(
            {
                'user': [],
                'item': [],
                'topk': [],
                'new_topk': [],  # Add new_topk to the results
                'counterfactual': [],
                'predicted_scores': [],
                'replacement': []
            }
        )
    return all_results

def append_result(ks, all_results, user_id, res):
    """
    append res to all_results where res is the result of an explanation algorithm
    :param ks: list of k values considered
    :param all_results: a dataset of results
    :param user_id: id of user explained
    :param res: the result produced by the explanation algorithms
    """
    for j in range(len(ks)):
        all_results[j]['user'].append(user_id)
        counterfactual, rec, topk, new_topk, predicted_scores, repl = res[j]
        all_results[j]['item'].append(rec)
        all_results[j]['topk'].append(topk)
        all_results[j]['new_topk'].append(new_topk)  # Append new_topk to the results
        all_results[j]['counterfactual'].append(counterfactual)
        all_results[j]['predicted_scores'].append(predicted_scores)
        all_results[j]['replacement'].append(repl)

        print('k =', ks[j])
        if not counterfactual:
            print(f"Can't find counterfactual set for user {user_id}")
        else:
            print(f"Found a set of size {len(counterfactual)}: {counterfactual}")
            print("Old top k: ", topk)
            print("New top k after replacement: ", new_topk)
            print("Replacement: ", repl, predicted_scores)


