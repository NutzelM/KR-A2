from typing import Union
from BayesNet import BayesNet
import numpy as np
import pandas as pd
from collections import defaultdict
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net


def min_degree_order(BN: BayesNet) -> list:
    """
    Computes the order of variables that needs to be followed for the summing-out 

    :param BN: BayesNet class
    :return: list of the order of variables
    """

    variables = BN.get_all_variables()
    graph = BN.get_interaction_graph()
    pi_order = []

    while len(variables) > 0:

        num_neighbors = dict(graph.degree(variables))
        min_pi = min(num_neighbors, key=num_neighbors.get)
        pi_order.append(min_pi)

        neighbors_pi = list(graph.neighbors(min_pi))

        for n_1 in neighbors_pi:
            for n_2 in neighbors_pi:
                if graph.number_of_edges(n_1, n_2) == 0 and n_1 != n_2:
                    graph.add_edge(n_1, n_2)
            
        graph.remove_node(min_pi)
        variables.remove(min_pi)

    return pi_order


def compute_marginals(order: list, BN: BayesNet) -> pd.DataFrame:
    """
    Computes the marginals for each variable by using multiplication and summing out

    :param order: the order for which all marginals are being computed
    :param BN: BayesNet class
    :return: pandas dataframes for each variable with their 'p' 
    """

    for pi in order:
        all_cpts = BN.get_all_cpts()    
        pi_cpt = all_cpts[pi]
        for var in all_cpts:
            list_of_vars = list(all_cpts[var].columns)
            if pi == list_of_vars[0] and pi != var and len(list_of_vars) > 2:
                
                # multiplication step
                cols = all_cpts[pi].columns[:-1].intersection(all_cpts[var].columns[:-1]).tolist()
                merge_vars = all_cpts[pi].merge(all_cpts[var], on=cols)
                multiply = merge_vars.assign(p = merge_vars.p_x*merge_vars.p_y).drop(columns=['p_x', 'p_y'])
                
                # summing out
                in_between_vars = list_of_vars[1:-1]
                summing = multiply.groupby(in_between_vars, as_index=False)['p'].sum()
       
                BN.update_cpt(var, summing)
    
    summing_out = BN.get_all_cpts()

    return summing_out

def merge_columns(df_summed_out: pd.DataFrame, Q: list):
    """
    Merges the columns for the variables Q by multiplying on each others marginals columns 

    :param df_summed_out: the computed  marginals for each variable Q
    :param Q: list of variables of interest
    :return: pandas dataframe with the merged multiplication column 'p' and each variable Q truth assignment column
    """

    # merge all columns 
    df_merged = df_summed_out[Q[0]]
    for i in range(len(Q)-1):
        df_merged = df_merged.merge(df_summed_out[Q[i+1]], how='cross')
    df_pior_marginal = df_merged[Q]
    df_values = df_merged.drop(columns = Q)
    
    # multiply all columns-values
    p_multiplied = df_values.iloc[:, 0]
    for i in range(len(df_values.columns)-1):
        p_multiplied *= df_values.iloc[:, i+1]
    df_pior_marginal['p'] = p_multiplied

    return df_pior_marginal


def prior_marginal(Q: list, BN: BayesNet):
    """
    Computes the prior marginal distribution for one or multiple variables.

    :param Q: list of variables of interest
    :param BN: BayesNet class
    :return: pandas dataframe with the computed marginals and truth values of each variable Q
    """

    #elimination order
    order = min_degree_order(BN)
  
    # computing procedure 
    summing_out = compute_marginals(order, BN)
    
    # if more variables are asked
    if len(Q) > 1:
        df_prior_marginal = merge_columns(summing_out, Q)
    
    # if one variable
    else:
        df_pior_marginal = summing_out[Q[0]]
    
    return df_pior_marginal

def posterior_marginal(E: pd.Series, Q: list, BN: BayesNet):
    """
    Computes the posterior marginal distribution for one or multiple variables.

    :param E: evidence. E.g.: pd.Series({"A", True}, {"B", False})
    :param Q: list of variables
    :param BN: BayesNet class
    :return: pandas dataframe with the computed marginals and truth values of each variable Q
    """
    E_key_list = []
    for key in sorted(E.keys()):
        E_key_list.append(key)

    all_cpts = BN.get_all_cpts()
    # reduce all cpts with factor E
    for key in all_cpts:
        cpt_recuded = BN.reduce_factor(E, all_cpts[key])
        # deletes cells with p = 0
        BN.update_cpt(key, cpt_recuded)

    # eliminate in min degree order 
    pi = min_degree_order(BN)
    summing_out = compute_marginals(pi, BN)
    # if more variables are asked
    if len(Q) > 1:
        df_post_marg = merge_columns(summing_out, Q)
    # if one variable
    else:
        df_post_marg = summing_out[Q[0]]
    
    # normalising
    # with more than 1 variable in evidence: 
    if len(E.index.values) > 1:
        df_pr = merge_columns(summing_out, E.index.values)  # merge columns of Pr 
        pr = df_pr[df_pr['p'] != 0]['p'].values[0]          # get Pr value of evidence
        df_post_marg['p'] = df_post_marg['p'] / pr          # normalise by Pr of evidence
    
    # only one variable in evidence:
    else:
        df_pr = summing_out[E.index.values[0]]              # merge columns of Pr 
        pr = df_pr[df_pr['p'] !=0]['p'].values              # get Pr value of evidence
        df_post_marg['p'] = df_post_marg['p'] / pr          # normalise by Pr of evidence

    return df_post_marg


if __name__ == '__main__':

    file_path = "testing/lecture_example3.BIFXML"
    network = BNReasoner(file_path)
    BN = network.bn

    #E = pd.Series({"Winter?": True, "Sprinkler?": False})
    E = pd.Series({"Winter?": True})
    Q = ['Rain?']

    if len(E) == 0:
        distribution = prior_marginal(Q, BN)
        print(distribution)
    else:
        distribution = posterior_marginal(E, Q, BN)
        print(distribution)

     

    



