from typing import Union
from BayesNet import BayesNet


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
file_path = "testing/lecture_example.BIFXML"
network = BNReasoner(file_path)
print(network)

    # TODO: This is where your methods should go
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


def min_degree_order(BN):

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


def compute_marginals(order):

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


def prior_marginal(Q, BN):
    
    #elimination order
    order = min_degree_order(BN)
  
    # computing procedure 
    summing_out = compute_marginals(order)
    
    if len(Q) > 1:
        df = summing_out[Q[0]]
        for i in range(len(Q)-1):
            df = df.merge(summing_out[Q[i+1]], how='cross')
        print(df)
    else:
        df = summing_out[Q[0]]
        print(df)
    
    return df



if __name__ == '__main__':

    file_path = "testing/lecture_example.BIFXML"
    network = BNReasoner(file_path)
    BN = network.bn

    evidence = []
    Q = ['Wet Grass?']

    if evidence == None or evidence == []:
        distribution = prior_marginal(Q, BN)

     

    



