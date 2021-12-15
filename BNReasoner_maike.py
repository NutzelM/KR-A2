from os import environb
from typing import Union, final
from networkx.algorithms.assortativity import pairs

from networkx.algorithms.dag import descendants
from numpy import multiply
from pandas.core.indexes.api import all_indexes_same
from pandas.core.series import Series
from BayesNet import BayesNet
import pandas as pd
from copy import deepcopy
from BayesNet import BayesNet
import numpy as np

#TODO: add node pruning for posterior etc, fix normalisation function do it sums out all the vars not in evidence (hence also adding the velues from the other ctps)
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

def delete_zero_rows(cpt):
    """
    deletes the rows where p = 0
    """
    # gets index with rows of p value 0
    indexzeros = cpt[ cpt['p'] == 0 ].index
    # deltes the index of the rows
    cpt = cpt.drop(indexzeros)
    return cpt

def reduce(evidence, cpt, key):
    cpt_reduced = BN.reduce_factor(evidence, cpt)
    cpt_recuced = delete_zero_rows(cpt_reduced)
    BN.update_cpt(key ,cpt_recuced)
    # drop columns all true/false
    return cpt_recuced

def delete_evidence_columns(cpt, evidence):
    evidence_vars = list(evidence.index.values)
    for v in evidence_vars:
        cpt.drop(v, axis=1, inplace=True)
    return cpt
def order(pi):
    return pi 

def min_degree_order(variables):

    graph = BN.get_interaction_graph()
    pi_order = []
    while len(variables) > 0:
        
        num_neighbors = dict(graph.degree(variables))
        min_pi = min(num_neighbors, key=num_neighbors.get)
        #print(f"the min pi is {min_pi}")
        pi_order.append(min_pi)

        neighbors_pi = list(graph.neighbors(min_pi))

        for n_1 in neighbors_pi:
            for n_2 in neighbors_pi:
                if graph.number_of_edges(n_1, n_2) == 0 and n_1 != n_2:
                    graph.add_edge(n_1, n_2)
            
        graph.remove_node(min_pi)
        variables.remove(min_pi)

    return pi_order

def normalise(cpt_to_be_normalised, evidence):
    all_cpts = BN.get_all_cpts()
    normalisation_value = 1
    for var in evidence.index.values:
        #print(f"len of cpts of {var} is {len(all_cpts[var])}")
        cpt = all_cpts[var]
        # if value is not in a 1x1 table, multiply sum out the other values
        if len(cpt) > 1: 
            keys_to_multiply = []
            for key in all_cpts:
                columns = list(all_cpts[key].index.column)[:-1]
                if var in columns:
                    keys_to_multiply.append(key)
            cpt = multiply_cpts(all_cpts, keys_to_multiply)
            for c in columns:
                cpt = sum_out(cpt, c) 
        normalisation_value = normalisation_value*cpt.values[0][-1]
    #print(f"normalisation value is {normalisation_value}")
    cpt_to_be_normalised['p'] = (1/normalisation_value)* cpt_to_be_normalised['p']
    return cpt_to_be_normalised

def multiply_cpts(all_cpts, keys_to_multiply):
    cpt = all_cpts[keys_to_multiply[0]]
    for i in range(1, len(keys_to_multiply)):
        cpt_new = all_cpts[keys_to_multiply[i]]
        #print(f"intersection of {list(cpt.columns[:-1])} and {list(cpt_new.columns[:-1])}")
        columns = cpt.columns[:-1].intersection(cpt_new.columns[:-1]).tolist()
        merge_vars = cpt.merge(cpt_new, on =columns ) 
        cpt = merge_vars.assign(p = merge_vars.p_x*merge_vars.p_y).drop(columns=['p_x', 'p_y'])
    return cpt

def max_out(cpt, var):
    #print(f"MAXING OUT {var}")
    staying_vars = list(cpt.columns.values)[:-1]
    staying_vars.remove(var)
    if len(cpt.index) > 2:
        new_cpt = cpt.sort_values(['p'], ascending=False).drop_duplicates(subset=staying_vars, keep='first')
    else: 
        new_cpt = cpt[cpt['p']== cpt['p'].max()]
    # updates evidence with the 

    return new_cpt  

def sum_out(cpt, var):
    #print(f"SUMMING OUT {var}")
    staying_vars = list(cpt.columns.values)[:-1]
    staying_vars.remove(var)
    #print(f"old cpt is \n {cpt}")
    cpt = cpt.groupby(staying_vars, as_index=False)['p'].sum()
    #print(f"new cpt is \n {cpt}")
    return cpt

def posteriour_marginal(Q, evidence): 
    all_cpts = BN.get_all_cpts()
    # reduce with evidence 
    # The reduce method also updates the BN.cpts   
    for key in all_cpts:
        all_cpts[key] = reduce(evidence, all_cpts[key], key)
    all_vars = BN.get_all_variables()
    # in order to compute P(Q, e)
    pi = list(set(all_vars) - set(Q))
    pi = min_degree_order(pi)
    #print(f"PI IS {pi}")
    for i in range(len(pi)):
        keys_to_multiply = []
        for key in all_cpts:
            collumn_names = list(all_cpts[key].columns.values)[:-1]
            if pi[i] in collumn_names:
                keys_to_multiply.append(key)
        #cpt = all_cpts['Sprinkler?']
        #print(f" keys to mult {keys_to_multiply}")
        cpt = multiply_cpts(all_cpts, keys_to_multiply)
        cpt = sum_out(cpt, pi[i])
        # delete all the used cpts
        all_cpts = delete_cpts(all_cpts, keys_to_multiply)
        # save the new cpt under the name of the last collumn (besides p)
        all_cpts[list(cpt.columns.values)[-2]] = cpt
    first_key = list(all_cpts.keys())[0]
    final_cpt = all_cpts[first_key]
    #print("multiply final cts ")
    if len(all_cpts) > 1:
        final_keys_to_multiply = [first_key]
        for key in all_cpts:
            if key != first_key:
                final_keys_to_multiply.append(key)
        final_cpt = multiply_cpts(all_cpts, final_keys_to_multiply)
    final_cpt = normalise(final_cpt, evidence)
    return final_cpt

def prior_marginal(Q): 
    all_cpts = BN.get_all_cpts()
    # reduce with evidence    
    all_vars = BN.get_all_variables()
    pi = list(set(all_vars) - set(Q))
    pi = min_degree_order(pi)
    for i in range(len(pi)):
        keys_to_multiply = []
        for key in all_cpts:
            collumn_names = list(all_cpts[key].columns.values)[:-1]
            if pi[i] in collumn_names:
                keys_to_multiply.append(key)
        #cpt = all_cpts['Sprinkler?']
        cpt = multiply_cpts(all_cpts, keys_to_multiply)
        cpt = sum_out(cpt, pi[i])
        # delete all the used cpts
        all_cpts = delete_cpts(all_cpts, keys_to_multiply)
        # save the new cpt under the name of the last collumn (besides p)
        all_cpts[list(cpt.columns.values)[-2]] = cpt
    first_key = list(all_cpts.keys())[0]
    final_cpt = all_cpts[first_key]
    
    if len(all_cpts) > 1:
        final_keys_to_multiply = [first_key]
        for key in all_cpts:
            if key != first_key:
                final_keys_to_multiply.append(key)
        final_cpt = multiply_cpts(all_cpts, keys_to_multiply)

    return final_cpt

def delete_cpts(all_cpts, del_vars):
    for v in del_vars:
        all_cpts.pop(v)
    return all_cpts

def print_cpts(all_cpts):
    for c in all_cpts:
        print(all_cpts[c])

def prune_edges(evidence) -> None:
    E =  evidence.index.values
    #print('E is ' + str(E))
    all_cpts = BN.get_all_cpts()
    for evi in E:
        neighbours = BN.get_children(evi)
       # #print(f"neighbours of {evi} are {list(neighbours)}")
        if len(list(neighbours)) > 0:
            for e in neighbours:
                BN.del_edge((evi,e))
                cpt_reduced = reduce(evidence, all_cpts[e], e)
                all_cpts[e] = cpt_reduced
    return all_cpts

def MPE(evidence):
    all_cpts = BN.get_all_cpts()
    all_cpts = prune_edges(evidence)
    # update cpt for interaction graph of min_degree_order
    for key in all_cpts:
        BN.update_cpt(key, all_cpts[key])
    
    Q = BN.get_all_variables()
    pi = min_degree_order(Q)
    
    #pi  = ['J', 'I', 'X', 'Y', 'O']
    # reduce with evidence    
    for key in all_cpts:
        all_cpts[key] = reduce(evidence, all_cpts[key], key)
    
    for i in range(len(pi)):
        keys_to_multiply = []
        for key in all_cpts:
            collumn_names = list(all_cpts[key].columns.values)[:-1]
            if pi[i] in collumn_names:
                keys_to_multiply.append(key)
        cpt = multiply_cpts(all_cpts, keys_to_multiply)
        cpt = max_out(cpt, pi[i])
        # delete all the used cpts
        all_cpts = delete_cpts(all_cpts, keys_to_multiply)
        # save the new cpt under the name of the last collumn (besides p)
        all_cpts[list(cpt.columns.values)[-2]] = cpt
    
    first_key = list(all_cpts.keys())[0]
    final_cpt = all_cpts[first_key]
    
    # multiply if multiple factos remail 
    if len(all_cpts) > 1:
        final_keys_to_multiply = [first_key]
        for key in all_cpts:
            if key != first_key:
                final_keys_to_multiply.append(key)
        final_cpt = multiply_cpts(all_cpts, final_keys_to_multiply)
    final_cpt = final_cpt[final_cpt['p']== final_cpt['p'].max()]
    return final_cpt

def MAP(M, evidence):
    all_cpts = BN.get_all_cpts()
    all_cpts = prune_edges(evidence)
    Q = BN.get_all_variables()
    #pi = min_degree_order(Q)
    pi  = ['O', 'Y', 'X', 'I', 'J']
    # reduce with evidence    
    for key in all_cpts:
        all_cpts[key] = reduce(evidence, all_cpts[key], key)
    
    for i in range(len(pi)):
        keys_to_multiply = []
        for key in all_cpts:
            collumn_names = list(all_cpts[key].columns.values)[:-1]
            if pi[i] in collumn_names:
                keys_to_multiply.append(key)
        cpt = multiply_cpts(all_cpts, keys_to_multiply)
        if pi[i] not in M:
            cpt = sum_out(cpt, pi[i])
        else:
            cpt = max_out(cpt, pi[i])
        # delete all the used cpts
        all_cpts = delete_cpts(all_cpts, keys_to_multiply)
        # save the new cpt under the name of the last collumn (besides p)
        all_cpts[list(cpt.columns.values)[-2]] = cpt
    
    first_key = list(all_cpts.keys())[0]
    final_cpt = all_cpts[first_key]
    
    # multiply if multiple factos remail 
    if len(all_cpts) > 1:
        final_keys_to_multiply = [first_key]
        for key in all_cpts:
            if key != first_key:
                final_keys_to_multiply.append(key)
        final_cpt = multiply_cpts(all_cpts, final_keys_to_multiply)
    
    final_cpt = final_cpt[final_cpt['p']== final_cpt['p'].max()]
    return final_cpt

if __name__ == '__main__':

    file_path = "testing/lecture_example.BIFXML"
    network = BNReasoner(file_path)
    BN = network.bn
    all_cpts = BN.get_all_cpts()
    evidence = pd.Series({'Winter?': True, 'Sprinkler?': False})
    M = ['I', 'J']
    Q =  ['Wet Grass?', 'Slippery Road?']
    all_cpts = BN.get_all_cpts()
    #print_cpts(all_cpts)
    #print(prior_marginal(Q))
    print(posteriour_marginal(Q, evidence))
    vars = BN.get_all_variables()
    #print(MPE(evidence))


