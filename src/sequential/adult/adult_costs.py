import numpy as np

from feature_cost_model.action_cost import ActionCost


class IncreaseAgeCosts(ActionCost):
    def __init__(self, features, dependency_graph=None):
        self.features = features
        feature_idx = self.features["age"]
        super().__init__(feature_idx, dependency_graph)

    def _get_costs(self, old_state, current_state):
        change_value = current_state[self.feature_idx] - old_state[self.feature_idx]
        assert (
            type(change_value) == float or type(change_value) == np.float64
        ), type(change_value)
        return abs(change_value)


class IncreaseCapitalGainCosts(ActionCost):
    def __init__(self, features, dependency_graph=None):
        self.features = features
        feature_idx = self.features["capital_gain"]
        super().__init__(feature_idx, dependency_graph)

    def _get_costs(self, old_state, current_state):
        change_value = abs(
            current_state[self.feature_idx] - old_state[self.feature_idx]
        )
        return change_value / 500


class IncreaseEducationCosts(ActionCost):
    def __init__(self, features, dependency_graph=None):
        self.features = features
        feature_idx = self.features["education"]
        self.general_costs = [
            0.0,  # nothing to School
            3.0,  # School to HS
            3.0,  # HS to college
            1.0,  # college to prof-school
            2.0,  # prof-school to assoc
            3.5,  # assoc to bachelors
            2.5,  # bachelors to masters
            5.0,  # masters to doctorate
        ]

        self.education_level_order = [0, 1, 2, 3, 4, 5, 6, 7]
        super().__init__(feature_idx, dependency_graph)

    def _get_costs(self, old_state, current_state):
        new_level = self.education_level_order.index(current_state[self.feature_idx])
        previous_level = self.education_level_order.index(old_state[self.feature_idx])

        if new_level == previous_level:
            return 0.0
        else:
            # return the cumulative costs to get that degree from the current one
            return float(sum(self.general_costs[previous_level + 1 : new_level + 1]))


class ChangeWorkHrsCosts(ActionCost):
    def __init__(self, features, dependency_graph=None):
        self.features = features
        feature_idx = self.features["hours_per_week"]
        super().__init__(feature_idx, dependency_graph)

    def _get_costs(self, old_state, current_state):
        change_value = current_state[self.feature_idx] - old_state[self.feature_idx]
        # discount based on direction, reducing is easier than increasing
        # TODO think about this
        discount = 1.0
        if change_value < 0.0:
            # reducing is free of costs
            discount = 0.0
        # # TODO 0.0
        # return 0.0
        return abs(change_value) * discount


class ChangeCategoricalCosts(ActionCost):
    def __init__(self, feature_idx, features, dependency_graph=None):
        self.features = features
        super().__init__(feature_idx, dependency_graph)

    def _get_costs(self, old_state, current_state):
        old_value = old_state[self.feature_idx]
        new_value = current_state[self.feature_idx]
        if old_value == new_value:
            return 0.0
        else:
            return 5.0