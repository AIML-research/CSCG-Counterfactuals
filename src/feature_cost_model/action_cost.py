import numpy as np
from feature_cost_model.feature_dependency_graph import FeatureDependencyGraph


class ActionCost:
    def __init__(self, feature_idx, dependency_graph=None):
        self.feature_idx = feature_idx
        self.G = dependency_graph
        if self.G is None:
            self.G = FeatureDependencyGraph(None)

    def get_costs(self, old_state, new_state):
        """Returns the costs based on this action and the current state

        Args:
            current_state (np.ndarray): Current tweaked instance state
            change_values (np.ndarray): All change value values with respect to the previous and current state
        """
        assert len(old_state) == len(new_state)

        base_cost = self._get_costs(old_state, new_state)
        assert type(base_cost) == float or type(base_cost) == np.float64, type(
            base_cost
        )
        assert base_cost >= 0.0, base_cost

        return base_cost

    def _get_costs(self, old_state, current_state):
        raise Exception("Not implemented")