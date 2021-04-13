import numpy as np


class FeatureDependencyGraph:
    """
    G = {
        (FeatA_idx: int, FeatB_idx: int): func(x: current_state),
        FeatA affects FeatB
    }
    """

    def __init__(self, G, debug=False):
        self.G = G
        self.incoming_edges = self.get_incoming_edge_function()
        self.debug = debug
        if self.debug:
            print(self)

    def __str__(self):
        if self.G is None:
            return "Empty graph"
        res = ""
        for edge in self.G.keys():
            A, B = edge
            res += f"{A}->{B}, "
        return res

    def get_incoming_edge_function(self):
        edges = {}
        if self.G is None:
            return edges
        for edge, func in self.G.items():
            from_node, to_node = edge
            if to_node in edges.keys():
                edges[to_node].append(func)
            else:
                edges[to_node] = [func]
        return edges

    def get_mean_discount_factor(self, feature_idx: int, current_state: np.ndarray):
        """Returns mean discount factor for the respective feature with respect
        to the connected features provided by the dependency graph

        Args:
            feature_idx (int): The feature idx for which the discount shall be copmputed
            current_state (np.ndarray): Current tweaked instance state (before applying the current action)

        Returns:
            float: Mean discount factor
        """
        if self.G is None:
            # If there is no graph, then there is no discount, i.e. discount = 1.0
            return 1.0
        if feature_idx not in self.incoming_edges.keys():
            return 1.0
        discounts = []
        for func in self.incoming_edges[feature_idx]:
            # try to apply the function. If the value is outside the range then
            # we penalize it by a high discount (will be handled later on in the penalty evaluation anyway,
            # so we don't need to care about it too much)
            try:
                discount = func(current_state)
            except:
                return 2.0
            discounts.append(discount)
            if self.debug:
                print(f"Edge discount factor to {feature_idx} is {discount}")
        assert np.nan not in discounts
        assert None not in discounts
        if self.debug:
            print(f"Mean discount is {np.mean(discounts)}")
        return np.mean(discounts)

    def get_mean_action_discount(self, current_state: np.ndarray, affected_features):
        """Returns the mean discount of all affected features by an action.
        I.e., the mean of the mean feature discounts.

        Args:
            current_state (np.ndarray): The current x state, before applying the action
            affected_features (list or np.ndarray): Affected feature indices

        Returns:
            float: Mean of mean feature discounts
        """
        if self.G is None:
            # If there is no graph, then there is no discount, i.e. discount = 1.0
            return 1.0
        assert (
            type(affected_features) == list or type(affected_features) == np.ndarray
        ), type(affected_features)
        assert len(affected_features) > 0, len(affected_features)
        assert all(
            [type(x) == int or type(x) == np.int64 for x in affected_features]
        ), affected_features
        assert all(
            [0 <= x < len(current_state) for x in affected_features]
        ), affected_features
        mean_feature_discounts = []
        for feature_idx in affected_features:
            if feature_idx in self.incoming_edges.keys():
                feature_discount = self.get_mean_discount_factor(feature_idx, current_state)
                mean_feature_discounts.append(feature_discount)
        if len(mean_feature_discounts) == 0:
            return 1.0
        return np.mean(mean_feature_discounts)

    def get_mean_state_discount(self, current_state: np.ndarray):
        if self.G is None:
            # If there is no graph, then there is no discount, i.e. discount = 1.0
            return 1.0
        assert current_state.dtype == np.float64, current_state.dtype
        assert current_state.ndim == 1, current_state.ndim
        discounts = []
        for feature_idx, _ in enumerate(current_state):
            discounts.append(self.get_mean_discount_factor(feature_idx, current_state))
        assert np.nan not in discounts
        assert None not in discounts
        return np.mean(discounts)

    def get_mean_sequence_discount(self, all_states):
        if self.G is None:
            # If there is no graph, then there is no discount, i.e. discount = 1.0
            return 1.0
        assert len(all_states) > 0
        assert type(all_states) == list, type(all_states)
        assert type(all_states[0]) == np.ndarray, type(all_states[0])
        if len(all_states) == 0:
            return 1.0
        discounts = []
        for current_state in all_states:
            discounts.append(self.get_mean_state_discount(current_state))
        assert np.nan not in discounts
        assert None not in discounts
        return np.mean(discounts)


if __name__ == "__main__":
    G = {
        (0, 1): lambda x: 0.5 if x[0] == 2 else 1.0,
        (2, 1): lambda x: 0.2 if x[2] == 0.3 else 1.0,
        (3, 1): lambda x: x[3],
    }

    dependency_graph = FeatureDependencyGraph(G, debug=True)
    print(dependency_graph)

    x0 = np.array([0, 1, 2, 3, 4, 5, 6])
    print(dependency_graph.get_mean_discount_factor(feature_idx=1, current_state=x0))

    x0 = np.array([2, 1, 0.3, 3, 4, 5, 6])
    print(dependency_graph.get_mean_discount_factor(feature_idx=1, current_state=x0))