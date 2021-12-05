from typing import Union

from networkx.algorithms.dag import descendants
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
BN = network.bn
#BN.draw_structure()
vars = BN.get_all_variables()
for x in vars:
    parents = BN.get_parents(x)
    descendants = BN.get_descendants(x, [])
    non_descendents = BN.get_non_descendents(x)
    print('---------------------------------')
    print(f'variabale {x}')
    print(f'parents are {parents}')
    print(f'children are {descendants}')
    print(f'non_descendents are {non_descendents}')
    print('---------------------------------')


print(BN.get_children(vars[0]))


#def dSep(X,Y,Z, G):

