from typing import Union
from BayesNet import BayesNet
import copy


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

    def is_dsep(self,X,Y,Z):
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

file_path = "testing/lecture_example.BIFXML"
network = BNReasoner(file_path)
print(network.is_dsep('Sprinkler?','Rain?',{'Winter?'}))
network.bn.draw_structure()
