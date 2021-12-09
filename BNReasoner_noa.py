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


    multiplication = defaultdict(list)

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
                
                multiplication[str(list(multiply.columns))].append(multiply)

                # summing out
                in_between_vars = list_of_vars[1:-1]
                summing = multiply.groupby(in_between_vars, as_index=False)['p'].sum()
                
                #summing_out[str(list(summing.columns)[0:-1])].append(summing)
                
                BN.update_cpt(var, summing)
    
    summing_out = BN.get_all_cpts()

    return multiplication, summing_out


def prior_marginal(Q, BN):
    
    #elimination order
    order = min_degree_order(BN)
    order = filter(lambda i: i not in Q, order)

    # computing procedure 
    multiplication, summing_out = compute_marginals(order)
    print(multiplication)
    print(summing_out)

    return 


if __name__ == '__main__':

    file_path = "testing/lecture_example.BIFXML"
    network = BNReasoner(file_path)
    BN = network.bn

    evidence = []
    Q = ['Wet Grass?', 'Slippery Road?']

    if evidence == None or evidence == []:
        distribution = prior_marginal(Q, BN)
        print(distribution)
     

    



