from os import environ
from typing import Union, final
from networkx.algorithms.assortativity import pairs

from networkx.algorithms.dag import descendants
from numpy import True_, multiply
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import all_indexes_same
from pandas.core.series import Series
from BNReasoner_main import MPE
from BayesNet import BayesNet
import pandas as pd
from copy import deepcopy
from BayesNet import BayesNet
import copy 
import time 
import random

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

    ##########################################################
    # d Seperation. returns True if X is dSep from Y by Z
    ##########################################################
    def is_dsep(self,X,Z,Y):
        """
        returns true if 
        """
        # Gather all nodes, whose descendants are in Z
        obs_ancestors = set()
        observations = copy.copy(Z)
        while len(observations) > 0:
            for parent in self.bn.get_parents(observations.pop()):
                obs_ancestors.add(parent)
        
        # Traversing all active paths from X, if we end up at Y => not d-sep
        departures = {(X,'up')}
        visited = set()
        while len(departures) > 0:
            (node, direction) = departures.pop()
            if (node, direction) not in visited:
                visited.add((node, direction))

                # check if we end up at Y
                if node not in Z and node == Y:
                    return False

                if direction == "up" and node not in Z:
                    for parent in self.bn.get_parents(node):
                        departures.add((parent, 'up'))
                    for child in self.bn.get_children(node):
                        departures.add((child, 'down'))

                elif direction == 'down':
                    if node not in Z:
                        for child in self.bn.get_children(node):
                            departures.add((child, 'down'))
                    # Checks for V-structure
                    if node in Z or node in obs_ancestors:
                        for parent in self.bn.get_parents(node):
                            departures.add((child, 'up'))
        return True

    ##########################################################
    # helpful methods for computation marginals, MPE, MEP. etc.
    ##########################################################
    def delete_zero_rows(self,cpt):
        """ 
        deletes the rows of cpt where p = zero
        parameters: cpt
        returns : cpt
        """
        # gets index with rows of p value 0
        indexzeros = cpt[ cpt['p'] == 0 ].index
        # deltes the index of the rows
        cpt = cpt.drop(indexzeros)
        return cpt

    def reduce(self, evidence, cpt, key):
        """ 
        simplifies cpt with evidence
        parameters: cpt
        returns : cpt
        """
        cpt_reduced = self.bn.reduce_factor(evidence, cpt)
        cpt_reduced = self.delete_zero_rows(cpt_reduced)
        self.bn.update_cpt(key ,cpt_reduced)
        # drop columns all true/false
        return cpt_reduced

    def normalise(self, cpt_to_be_normalised, evidence):
        """ 
        normalises a cpt based on the evidence to go from P(Q,e) to P(Q|e)
        parameters: cpt: cpt to be normalised, evidence: evidence to normalise with
        returns : cpt 
        """
        all_cpts = self.bn.get_all_cpts()
        normalisation_value = 1
        # for all the variables in the evidence
        for var in evidence.index.values:
            cpt = all_cpts[var]
            # if value is not in a 1x1 table, mutiply 
            if len(cpt) > 1: 
                keys_to_multiply = self.find_keys_to_multiply(all_cpts, var)
                cpt = self.multiply_cpts(all_cpts, keys_to_multiply)
            normalisation_value = normalisation_value*cpt.values[0][-1]
        cpt_to_be_normalised['p'] = (1/normalisation_value)* cpt_to_be_normalised['p']
        return cpt_to_be_normalised

    def multiply_cpts(self, all_cpts, keys_to_multiply):
        """ 
        multiplies all the cpts with names in given list
        parameters: 
        all_cpts: a dictionary of all cps, 
        keys to multiply: list of names of cpts to be multiplied
        returns : cpt 
        """
        cpt = all_cpts[keys_to_multiply[0]]
        new_keys_to_multiply = [keys_to_multiply[0]]
        for i in range(1, len(keys_to_multiply)):
            cpt_new = all_cpts[keys_to_multiply[i]]
            columns = cpt.columns[:-1].intersection(cpt_new.columns[:-1]).tolist()
            # only merge if the cpts have a common variable (for product at the end)
            if len(columns) > 0:
                merge_vars = cpt.merge(cpt_new, on = columns) 
                cpt = merge_vars.assign(p = merge_vars.p_x*merge_vars.p_y).drop(columns=['p_x', 'p_y'])
                all_cpts[keys_to_multiply[0]] = cpt
            else:
                new_keys_to_multiply.append(keys_to_multiply[i])
        # in the case of a merging error (for product at the end)
        if len(new_keys_to_multiply) > 1:
            cpt = self.multiply_cpts(all_cpts, new_keys_to_multiply)
        return cpt

    def max_out(self, cpt, var):
        """ 
        maxes out given variable to a cpt and return that cpt
        parameters: cpt, var: variable that will be maxed out
        returns : cpt 
        """
        staying_vars = list(cpt.columns.values)[:-1]
        staying_vars.remove(var)
        if len(cpt.index) > 2:
            new_cpt = cpt.sort_values(['p'], ascending=False).drop_duplicates(subset=staying_vars, keep='first')
        else: 
            new_cpt = cpt[cpt['p']== cpt['p'].max()] 
        return new_cpt  

    def sum_out(self, cpt, var):
        """ 
        sums out given variable to a cpt and return that cpt
        parameters: cpt, var: variable that will summed out
        returns : cpt 
        """
        staying_vars = list(cpt.columns.values)[:-1]
        staying_vars.remove(var)
        cpt = cpt.groupby(staying_vars, as_index=False)['p'].sum()
        return cpt

    def find_keys_to_multiply(self, all_cpts, var):
        """ 
        finds all cpts that have var in collumn and puts in list
        parameters: all_cpts: dictionary of all cpts, var: the variable to find matches with
        returns : list keys_to_multiply of all vars to multipy
        """
        keys_to_multiply = []
        for key in all_cpts:
            # list of all names in the collumns
            collumn_names = list(all_cpts[key].columns.values)[:-1]
            if var in collumn_names:
                keys_to_multiply.append(key)
        return keys_to_multiply

    def delete_cpts(self, all_cpts, del_vars):
        """ 
        computes prior marginal of P(Q) (by just calling posteriour_marginal with empty evidence)
        parameters: Q: list of variables 
        returns : cpt of P(Q)
        """
        for v in del_vars:
            all_cpts.pop(v)
        return all_cpts

    def print_cpts(self, all_cpts) -> None:
        """ 
        prints all cpts
        parameters: all_cpts: dictionary of all cpts to print
        """
        for c in all_cpts:
            print(all_cpts[c])

    #############################
    # Network pruning methods
    #############################
    def prune_edges(self, all_cpts, evidence):
        """ 
        eliminates edges e -> x and replaces cpt of x with reduced version
        parameters: Q: list of variables 
        returns : all_cpts: dictionary of all cpts
        """
        E =  evidence.index.values
        for evi in E:
            neighbours = self.bn.get_children(evi)
            if len(list(neighbours)) > 0:
                for e in neighbours:
                    self.bn.del_edge((evi,e))
                    cpt_reduced = self.sum_out(all_cpts[e], evi)
                    cpt_reduced = self.reduce(evidence, all_cpts[e], e)
                    all_cpts[e] = cpt_reduced
        return all_cpts

    def prune_nodes(self, all_cpts, evidence, Q):
        """ 
        eliminates leaf nodes not in Q or E
        parameters: Q: list of variables 
        returns : all_cpts: dictionary of all cpts
        """
        E =  evidence.index.values
        pruned_bool = False
        for key in all_cpts:
            # if variable is not in E and has no children (is leaf)
            if key not in E and key not in Q and BN.get_children(key) == None:
                BN.del_var(key)
                all_cpts.pop(key)
                pruned_bool = True
        
        if not pruned_bool:
            return all_cpts
        # keep going recurively untill pruning no longer possible
        return self.prune_nodes(evidence, Q)

    def prune_network(self, all_cpts, evidence,  Q):           
        """ 
        node and edge prunes a network
        returns : all_cpts
        """
        all_cpts_new = self.prune_nodes(all_cpts, evidence, Q)
        all_cpts_new = self.prune_edges(all_cpts, evidence)
        # pruning can be done iteratively untill no further changes
        for key in all_cpts_new:
            if not all_cpts_new[key].equals(all_cpts[key]):
                return self.prune_network(all_cpts_new, evidence, Q)
        return all_cpts_new

    #############################
    # Ordering methods
    #############################
    def order(self, variables, ordering):
        """
        orders variables with the chosen method
        parameter ordering: method of choice
        returns: ordered variables po
        """
        random.shuffle(variables)
        if ordering == 'min_degree': 
            return  self.min_degree_order(variables)
        if ordering == 'min_fill': 
            return  self.min_fill_order(variables)
        if ordering == 'random':
            return variables
        print("no valid ordering method has been chosen, returning same order")
        return variables

    def min_degree_order(self, variables):
        """ 
        orders list of variables based on minimal degree
        parameters: list of variables
        returns : pi: ordered list of variables
        """
        graph = self.bn.get_interaction_graph()
        pi_order = []
        while len(variables) > 0:
            
            num_neighbors = dict(graph.degree(variables))
            min_pi = min(num_neighbors, key=num_neighbors.get)
            pi_order.append(min_pi)

            neighbors_pi = list(graph.neighbors(min_pi))

            for n_1 in neighbors_pi:
                for n_2 in neighbors_pi:
                    # not adjacent neighbours and not itsself
                    if graph.number_of_edges(n_1, n_2) == 0 and n_1 != n_2:
                        graph.add_edge(n_1, n_2)     
            graph.remove_node(min_pi)
            variables.remove(min_pi)
        return pi_order

    def min_fill_order(self, variables):
        """ 
        orders list of variables based on node whose elimination adds the smallest number of degrees/
        parameters: list of variables
        returns : pi: ordered list of variables
        """
        graph = self.bn.get_interaction_graph()
        pi_order = []
        variables_dict = dict.fromkeys(variables, 0)
        while len(variables) > 0:
            # find the variable in variables where the most amount of edges have to be reated if it was removed
            for var1 in variables:
                neighbors_var1 = list(graph.neighbors(var1))
                for n_1 in neighbors_var1:
                    for n_2 in neighbors_var1:
                        if graph.number_of_edges(n_1, n_2) == 0 and n_1 != n_2:
                            # ann edge has to be added
                            variables_dict[var1] += 1  
            # get key where the value is minimal (least amount of edges were added)
            min_pi = min(variables_dict, key=variables_dict.get)
            pi_order.append(min_pi)
            neighbors_pi = list(graph.neighbors(min_pi))
            for n_1 in neighbors_pi:
                for n_2 in neighbors_pi:
                    # not adjacent neighbours and not itsself
                    if graph.number_of_edges(n_1, n_2) == 0 and n_1 != n_2:
                        graph.add_edge(n_1, n_2)   
            graph.remove_node(min_pi)
            variables.remove(min_pi)
            variables_dict.pop(min_pi)
        return pi_order


    #################################
    # compute marginals, MPE, MEP
    #################################
    def prior_marginal(self, Q, ordering): 
        """ 
        computes prior marginal of P(Q) (by just calling posteriour_marginal with empty evidence)
        parameters: Q: list of variables 
        returns : cpt of P(Q)
        """
        # prior marginal is just posterior_marginal with an empty evidence.
        return self.posteriour_marginal(Q, pd.Series({}), ordering)

    def posteriour_marginal(self, Q, evidence, ordering): 
        """ 
        computes posteriour marginal of P(Q|evidence)
        parameters: Q: list of variables , evidence: Series of {variable: True/False}
        returns : cpt of P(Q|evidence)
        """
        all_cpts = self.bn.get_all_cpts()
        # reduce with evidence 
        # The reduce method also updates the BN.cpts   
        for key in all_cpts:
            all_cpts[key] = self.reduce(evidence, all_cpts[key], key)
        all_vars = self.bn.get_all_variables()
        # pi: all elements not in Q
        pi = list(set(all_vars) - set(Q))
        # order the pi with heuristic
        pi = self.order(pi, ordering)
        # factor and sum out all the elements not in Q
        for i in range(len(pi)): 
            # match pi[i] to all cpts that contain pi[i]
            keys_to_multiply = self.find_keys_to_multiply(all_cpts, pi[i])
            # multiply all these cpts
            cpt = self.multiply_cpts(all_cpts, keys_to_multiply)
            # sum out cpts
            cpt = self.sum_out(cpt, pi[i])
            # delete all the used cpts")
            all_cpts = self.delete_cpts(all_cpts, keys_to_multiply)
            # save the new cpt under the name of the last collumn (besides p)
            all_cpts[pi[i]] = cpt
        
        first_key = list(all_cpts.keys())[0]
        final_cpt = all_cpts[first_key]
        

        #  multiply remaining
        if len(all_cpts) > 1:
            final_cpt = self.multiply_cpts(all_cpts, list(all_cpts.keys()))
        
        final_cpt = self.normalise(final_cpt, evidence)
        
        #delete (sum out) evidence columns not in Q
        for c in list(final_cpt.columns.values)[:-1]:
            if c not in Q:
                final_cpt = self.sum_out(final_cpt, c)
        return final_cpt

    def MPE(self, evidence, ordering):
        start = time.time()
        """ 
        computes most likely instantaion based on evidence (by variable elimination)
        parameter evidence: series of evidence
        parameter ordering: choice of ordering method, String
        returns : cpt of MPE probability
        """
        all_cpts = self.bn.get_all_cpts()
        all_cpts = self.prune_edges(all_cpts, evidence)
        # update cpt for interaction graph of min_degree_order
        for key in all_cpts:
            self.bn.update_cpt(key, all_cpts[key])

        Q = self.bn.get_all_variables()
        pi = self.order(Q, ordering)
        # reduce with evidence    
        for key in all_cpts:
            all_cpts[key] = self.reduce(evidence, all_cpts[key], key)
        
        for i in range(len(pi)):
            print(time.time() - start)
            if (time.time() - start) >= 700:
                return False
            # match pi[i] to all cpts that contain pi[i]
            keys_to_multiply = self.find_keys_to_multiply(all_cpts, pi[i])
            # multiply all these cpts
            cpt = self.multiply_cpts(all_cpts, keys_to_multiply)
            cpt = self.max_out(cpt, pi[i])
            # delete all the used cpts
            all_cpts = self.delete_cpts(all_cpts, keys_to_multiply)
            # save the new cpt under the name of the last collumn (besides p)
            all_cpts[pi[i]] = cpt
        
        final_cpt = all_cpts[list(all_cpts.keys())[0]]
        #  multiply remaining
        if len(all_cpts) > 1:
            final_cpt = self.multiply_cpts(all_cpts, list(all_cpts.keys()))

        final_cpt = final_cpt[final_cpt['p']== final_cpt['p'].max()]
        return final_cpt

    def MAP(self, M, evidence, ordering):
        """ 
        like MPE but first sums out non-MAP variables then maximize out MAP variables
        parameter M: MAP variables
        parameter evidence: series of evidence
        parameter ordering: choice of ordering method, String
        returns : cpt of MAP probability
        """
        all_cpts = self.bn.get_all_cpts()
        all_cpts = self.prune_network(all_cpts, evidence, [])
        Q = self.bn.get_all_variables()
        pi = self.order(Q, ordering)
        pi_ordered_M = []
        for var in pi:
            if var not in M:  pi_ordered_M.append(var)
        for var in M:
            pi_ordered_M.append(var)
        pi = pi_ordered_M
        # reduce with evidence    
        for key in all_cpts:
            all_cpts[key] = self.reduce(evidence, all_cpts[key], key)
        
        for i in range(len(pi)):
            # match pi[i] to all cpts that contain pi[i]
            keys_to_multiply = self.find_keys_to_multiply(all_cpts, pi[i])
            # multiply all these cpts
            cpt = self.multiply_cpts(all_cpts, keys_to_multiply)
            if pi[i] in M:
                cpt = self.max_out(cpt, pi[i])
            else:
                cpt = self.sum_out(cpt, pi[i])
            # delete all the used cpts
            all_cpts = self.delete_cpts(all_cpts, keys_to_multiply)
            # save the new cpt under the name of the last collumn (besides p)
            all_cpts[pi[i]] = cpt
        
        final_cpt = all_cpts[list(all_cpts.keys())[0]]
        #  multiply remaining
        if len(all_cpts) > 1:
            final_cpt = self.multiply_cpts(all_cpts, list(all_cpts.keys()))
            
        final_cpt = final_cpt[final_cpt['p']== final_cpt['p'].max()]
        return final_cpt

if __name__ == '__main__':

    file_path = "testing/lecture_example.BIFXML"
    network = BNReasoner(file_path)
    BN = network.bn
    #evidence = pd.Series({},dtype= object)
    
    M = []
    Q =  ['a']
    #evidence = pd.Series({'Winter?': False, 'Rain?' : True})
    evidence = pd.Series({'b': False, 'c' : True})
    possible_orderings = ['min_degree', 'min_fill', 'random']
    #print(network.MPE(evidence, possible_orderings[0]))
    #print(network.MPE(evidence, possible_orderings[0]))
    MPE_dataframe = None 
    for i in range(10):
        if i > 0:
            old_MPE_dataframe = MPE_dataframe
        MPE_2d_list = []
        for i in range(10, 101, 10):
            print(i)
            file_path = f"testing/network-{i}.BIFXML"
            network = BNReasoner(file_path)
            MPE_1d_list = []
            for order in possible_orderings:
                start = time.time()
                result = network.MPE(evidence, order)
                end = time.time()
                print(f"appending {end - start} to columns {order}")
                MPE_1d_list.append(end-start)
            MPE_2d_list.append(MPE_1d_list)
        print(MPE_2d_list)
        MPE_2d_list.append(list(range(10,100,10)))
        MPE_dataframe = pd.DataFrame(MPE_2d_list, columns = ['number of variables' ,'min degree heuristic', 'min fill heurisic', 'random (no heuristic'])
        if i > 0:
            MPE_dataframe = (MPE_dataframe + old_MPE_dataframe)/2
        print(MPE_dataframe)
    
    
