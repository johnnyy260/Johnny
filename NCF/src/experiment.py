from get_counterfactual import generate_cf
from helper import parse_args


def main():
    """
    Run the full experiment for the Accent algorithm
    The ks values represents the number of items in the topk that should be considered when finding counterfactuals.
    """
    # Parse command line arguments using the function parse_args from helper.py
    args = parse_args()

    # Define the list of k values to be used in the experiment
    ks = [20, 50, 100]
    
    # Call generate_cf function from get_counterfactual.py to generate counterfactual explanations 
    # for each k value using the algorithm specified in the command line arguments
    generate_cf(args.algo, ks)


if __name__ == "__main__":
    # Call the main function when the script is run
    main()