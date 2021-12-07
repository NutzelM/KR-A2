from typing import Union

from networkx.algorithms.dag import descendants
from BayesNet import BayesNet
import pandas as pd

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
BN = network.bn
#BN.draw_structure()
vars = BN.get_all_variables()
for x in vars:
    parents = BN.get_parents(x)
    descendants = BN.get_descendants(x, [])
    non_descendents = BN.get_non_descendents(x)
    # print('---------------------------------')
    # print(f'variabale {x}')
    # print(f'parents are {parents}')
    # print(f'children are {descendants}')
    # print(f'non_descendents are {non_descendents}')
    # print('---------------------------------')

#print(BN.get_all_cpts()['Slippery Road?'])
#print(BN.get_compatible_instantiations_table(pd.Series(("Winter?", False), ("Rain", True)), BN.get_all_cpts()['Rain?']))
all_cpts = BN.get_all_cpts()
all_vars = all_cpts.keys()
# for v in all_vars:
    
#     print(f"for variable {v} the cpt is")
#     df = all_cpts[v]
#     print(df)
#     print(".............")
#BN.summing_out('Winter?') 
# all_cpts = BN.get_all_cpts()
# all_vars = all_cpts.keys()
# for v in all_vars:
#     if v == 'Wet Grass?' or v == 'Rain?' or v == 'Sprinkler?':
#         print(f"for variable {v} the cpt is")
#         df = all_cpts[v]
#         print(df)
#     print(".............")
#BN.summing_out('Sprinkler?') 
# all_cpts = BN.get_all_cpts()
# all_vars = all_cpts.keys()
# for v in all_vars:
#     if v == 'Wet Grass?' or v == 'Rain?' or v == 'Sprinkler?':
#         print(f"for variable {v} the cpt is")
#         df = all_cpts[v]
#         print(df)
#     print(".............")
BN.posteriour_marginal(pd.Series({'Winter?' : True}), ['Rain?'])

#def dSep(X,Y,Z, G):

