from os import name
from typing import List, Tuple, Dict
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.dag import descendants
from numpy.core.fromnumeric import var
from pandas.core.frame import DataFrame
from pgmpy.readwrite import XMLBIFReader
import math
import itertools
import pandas as pd
import numpy as np
from copy import deepcopy
import operator 


class BayesNet:

    def __init__(self) -> None:
        # initialize graph structure
        self.structure = nx.DiGraph()

    # LOADING FUNCTIONS ------------------------------------------------------------------------------------------------
    def create_bn(self, variables: List[str], edges: List[Tuple[str, str]], cpts: Dict[str, pd.DataFrame]) -> None:
        """
        Creates the BN according to the python objects passed in.
        
        :param variables: List of names of the variables.
        :param edges: List of the directed edges.
        :param cpts: Dictionary of conditional probability tables.
        """
        # add nodes
        [self.add_var(v, cpt=cpts[v]) for v in variables]

        # add edges
        [self.add_edge(e) for e in edges]

        # check for cycles
        if not nx.is_directed_acyclic_graph(self.structure):
            raise Exception('The provided graph is not acyclic.')

    def load_from_bifxml(self, file_path: str) -> None:
        """
        Load a BayesNet from a file in BIFXML file format. See description of BIFXML here:
        http://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/

        :param file_path: Path to the BIFXML file.
        """
        # Read and parse the bifxml file
        with open(file_path) as f:
            bn_file = f.read()
        bif_reader = XMLBIFReader(string=bn_file)

        # load cpts
        cpts = {}
        # iterating through vars
        for key, values in bif_reader.get_values().items():
            values = values.transpose().flatten()
            n_vars = int(math.log2(len(values)))
            worlds = [list(i) for i in itertools.product([False, True], repeat=n_vars)]
            # create empty array
            cpt = []
            # iterating through worlds within a variable
            for i in range(len(values)):
                # add the probability to each possible world
                worlds[i].append(values[i])
                cpt.append(worlds[i])

            # determine column names
            columns = bif_reader.get_parents()[key]
            columns.reverse()
            columns.append(key)
            columns.append('p')
            cpts[key] = pd.DataFrame(cpt, columns=columns)
        
        # load vars
        variables = bif_reader.get_variables()
        
        # load edges
        edges = bif_reader.get_edges()

        self.create_bn(variables, edges, cpts)

    # METHODS THAT MIGHT ME USEFUL -------------------------------------------------------------------------------------

    def get_children(self, variable: str) -> List[str]:
        """
        Returns the children of the variable in the graph.
        :param variable: Variable to get the children from
        :return: List of children
        """
        return [c for c in self.structure.successors(variable)]
   
    def get_parents(self, variable: str) -> List[str]:
        """
        Returns the parents of the variable in the graph.
        :param variable: Variable to get the parents from
        :return: List of parents
        """
        return [c for c in self.structure.predecessors(variable)]
    
   
    def get_descendants(self, variable: str, descendants: List) -> List[str]:
        """
        Returns the descendents of the variable in the graph.
        :param variable: Variable to get the parents from
        :param descendants: list of descendents of Variable
        :return: List of descendents
        """
        children = self.get_children(variable)
        for child in children:
            if child not in descendants : descendants.append(child)
            self.get_descendants(child, descendants)
        return descendants

    
    def get_non_descendents(self, variable: str) -> List[str]:
        """
        Returns the non_descendents of the variable in the graph.
        :param variable: Variable to get the non decendents from
        :return: List of non decendents
        """
        parents = self.get_parents(variable)
        descendants = self.get_descendants(variable, [])
        vars = self.get_all_variables()
        return [c for c in vars if c not in (descendants or parents) if c != variable] 
   
   
    def get_cpt(self, variable: str) -> pd.DataFrame:
        """
        Returns the conditional probability table of a variable in the BN.
        :param variable: Variable of which the CPT should be returned.
        :return: Conditional probability table of 'variable' as a pandas DataFrame.
        """
        try:
            return self.structure.nodes[variable]['cpt']
        except KeyError:
            raise Exception('Variable not in the BN')

    def get_all_variables(self) -> List[str]:
        """
        Returns a list of all variables in the structure.
        :return: list of all variables.
        """
        return [n for n in self.structure.nodes]

    def get_all_cpts(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary of all cps in the network indexed by the variable they belong to.
        :return: Dictionary of all CPTs
        """
        cpts = {}
        for var in self.get_all_variables():
            cpts[var] = self.get_cpt(var)

        return cpts

    def get_interaction_graph(self):
        """
        Returns a networkx.Graph as interaction graph of the current BN.
        :return: The interaction graph based on the factors of the current BN.
        """
        # Create the graph and add all variables
        int_graph = nx.Graph()
        [int_graph.add_node(var) for var in self.get_all_variables()]

        # connect all variables with an edge which are mentioned in a CPT together
        for var in self.get_all_variables():
            involved_vars = list(self.get_cpt(var).columns)[:-1]
            for i in range(len(involved_vars)-1):
                for j in range(i+1, len(involved_vars)):
                    if not int_graph.has_edge(involved_vars[i], involved_vars[j]):
                        int_graph.add_edge(involved_vars[i], involved_vars[j])
        return int_graph
    
    def draw_interaction(self, Graph) -> None:
        """
        Visualize structure of the Graph.
        """
        nx.draw(Graph, with_labels=True, node_size=3000)
        plt.show()

    def min_degree_order(self):
        """
        Returns a oder in elimination order.
        :return: ordering pi of variables 
        """
        pi_list = []
        all_vars = self.get_all_variables()
        G = self.get_interaction_graph()
        # variable with least amount of neighbours
        while len(all_vars) > 0:
            pi = min(G.degree(all_vars), key=operator.itemgetter(1))[0]
            pi_list.append(pi)
            # all the neigbours of var_min_neighvours
            neighbours_pi = list(G.neighbors(pi))
            # add edge between non-ajacent neighbours
            for n in neighbours_pi:
                for n2 in neighbours_pi:
                    if G.number_of_edges(n,n2) == 0 and n is not n2:
                        G.add_edge(n, n2)
            G.remove_node(pi)
            all_vars.remove(pi)
            #self.draw_interaction(G)
        return pi_list


    def get_neighbors_from_int_graph(self, graph, variable):
        """
        Returns the neighbors from the interaction graph of a given variable.
        :return: The variable neighbors from the given variable based on the constructed interaction graph.
        """
        neighbors = nx.neighbors(graph, variable)
    
        return neighbors
        
        


    def summing_out(self, variable) -> None:
        """
        Sums out variable of all the tables in all_cpts and updates the table
        :param: variable: the variable to be summed out
        :return: None.
        """
        all_cpts = self.get_all_cpts()
        variable_cpt = all_cpts[variable]
        for key in all_cpts:
            # if the variable is part of conditional probability but not cps of variable itsself 
            list_of_vars = list(all_cpts[key].columns)
            if variable == list_of_vars[0] and variable != key and len(list_of_vars) > 2:
                cpt = all_cpts[key]
                variable_true_table = variable_cpt.loc[variable_cpt[variable] == True]
                variable_true_value = float(variable_true_table['p'].values)
                variable_false_table = variable_cpt.loc[variable_cpt[variable] == False]
                variable_false_value = float(variable_false_table['p'].values)
                # mutiplies factors
                cpt['p'] = cpt['p'] * np.where(cpt[variable] == True, variable_true_value, variable_false_value)
                # list of variables that should stay in table
                in_between_vars = list_of_vars[1:-1]
                # summing out variable
                cpt = cpt.groupby(in_between_vars, as_index=False)['p'].sum()
                # update dictionary with new table
                self.update_cpt(key, cpt)
    
    def prior_marginal(self, Q):
        """
        computes the prior marginal P(Q)
        :param: Q: subset of variables
        :return: None.
        """

        all_cpts = self.get_all_cpts()
        # eliminate in min degree order 
        pi = self.min_degree_order()
        for x in pi:
            self.summing_out(x)
        all_cpts = self.get_all_cpts()
        #TODO: combine relevant tables
        for key in all_cpts:
            if key in Q :
                print(all_cpts[key])
    
    def posteriour_marginal(self, E, Q) -> None:
        """
        Sums out variable of all the tables in all_cpts and updates the table
        :param: E: evidence
        :param: Q: subset of variables
        :return: None.
        """
        E_key_list = []
        for key in sorted(E.keys()):
            E_key_list.append(key)
            
        all_cpts = self.get_all_cpts()
        # reduce all cpts with factor E
        for key in all_cpts:
            cpt_recuded = self.reduce_factor(E, all_cpts[key])
            # deletes cells with p = 0
            self.update_cpt(key, cpt_recuded)
        # eliminate in min degree order 
        pi = self.min_degree_order()
        for x in pi:
            self.summing_out(x)
        all_cpts = self.get_all_cpts()
        #TODO: combine relevant tables
        for key in all_cpts:
            if key in Q :
                print(all_cpts[key])

     
    @staticmethod
    def get_compatible_instantiations_table(instantiation: pd.Series, cpt: pd.DataFrame):
        """
        Get all the entries of a CPT which are compatible with the instantiation.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series(("A", True), ("B", False))
        :param cpt: cpt to be filtered
        :return: table with compatible instantiations and their probability value
        """
        var_names = instantiation.index.values
        var_names = [v for v in var_names if v in cpt.columns]  # get rid of excess variables names
        compat_indices = cpt[var_names] == instantiation[var_names].values
        compat_indices = [all(x[1]) for x in compat_indices.iterrows()]
        compat_instances = cpt.loc[compat_indices]
        return compat_instances

    def update_cpt(self, variable: str, cpt: pd.DataFrame) -> None:
        """
        Replace the conditional probability table of a variable.
        :param variable: Variable to be modified
        :param cpt: new CPT
        """
        self.structure.nodes[variable]["cpt"] = cpt

    @staticmethod
    def reduce_factor(instantiation: pd.Series, cpt: pd.DataFrame) -> pd.DataFrame:
        """
        Creates and returns a new factor in which all probabilities which are incompatible with the instantiation
        passed to the method to 0.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series({"A", True}, {"B", False})
        :param cpt: cpt to be reduced
        :return: cpt with their original probability value and zero probability for incompatible instantiations
        """
        var_names = instantiation.index.values
        var_names = [v for v in var_names if v in cpt.columns]  # get rid of excess variables names
        if len(var_names) > 0:  # only reduce the factor if the evidence appears in it
            new_cpt = deepcopy(cpt)
            incompat_indices = cpt[var_names] != instantiation[var_names].values
            incompat_indices = [any(x[1]) for x in incompat_indices.iterrows()]
            new_cpt.loc[incompat_indices, 'p'] = 0.0
            return new_cpt
        else:
            return cpt

    def draw_structure(self) -> None:
        """
        Visualize structure of the BN.
        """
        nx.draw(self.structure, with_labels=True, node_size=3000)
        plt.show()

    # BASIC HOUSEKEEPING METHODS ---------------------------------------------------------------------------------------

    def add_var(self, variable: str, cpt: pd.DataFrame) -> None:
        """
        Add a variable to the BN.
        :param variable: variable to be added.
        :param cpt: conditional probability table of the variable.
        """
        if variable in self.structure.nodes:
            raise Exception('Variable already exists.')
        else:
            self.structure.add_node(variable, cpt=cpt)

    def add_edge(self, edge: Tuple[str, str]) -> None:
        """
        Add a directed edge to the BN.
        :param edge: Tuple of the directed edge to be added (e.g. ('A', 'B')).
        :raises Exception: If added edge introduces a cycle in the structure.
        """
        if edge in self.structure.edges:
            raise Exception('Edge already exists.')
        else:
            self.structure.add_edge(edge[0], edge[1])

        # check for cycles
        if not nx.is_directed_acyclic_graph(self.structure):
            self.structure.remove_edge(edge[0], edge[1])
            raise ValueError('Edge would make graph cyclic.')

    def del_var(self, variable: str) -> None:
        """
        Delete a variable from the BN.
        :param variable: Variable to be deleted.
        """
        self.structure.remove_node(variable)

    def del_edge(self, edge: Tuple[str, str]) -> None:
        """
        Delete an edge form the structure of the BN.
        :param edge: Edge to be deleted (e.g. ('A', 'B')).
        """
        self.structure.remove_edge(edge[0], edge[1])
