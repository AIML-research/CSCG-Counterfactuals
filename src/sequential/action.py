# from typing import Callable
import numpy as np
import pandas as pd

infty = 1e12


class Action:
    def __init__(
        self,
        action_id: int,
        key: int,
        feature_idx: int,
        initial_instance: np.ndarray,
        action_name: str,
        is_categorical,
        problem,
        dependency_graph=None,
        description="",
        feature_name=None,
        round_costs=False,
        print_application=False,
    ):
        self.action_id = action_id
        self.feature_idx = feature_idx
        self.action_name = action_name

        self.problem = problem

        self.key = key

        self.initial_instance = initial_instance
        self.initial_value = initial_instance[feature_idx]

        self.feature_name = feature_name or feature_idx

        self.is_categorical = is_categorical

        self.dependency_graph = dependency_graph

        self.round_costs = round_costs

        self.log = print_application

    def get_changed_values(self, old_instance, new_instance):
        assert old_instance.ndim == 1, old_instance.ndim
        assert old_instance.ndim == new_instance.ndim
        assert len(old_instance) == len(new_instance)
        assert old_instance.dtype == np.float64, old_instance.dtype
        assert new_instance.dtype == old_instance.dtype, (
            new_instance.dtype,
            old_instance.dtype,
        )
        change_values = []
        changed = []
        i = 0
        for old_val, new_val in zip(old_instance, new_instance):
            assert type(old_val) == float or type(old_val) == np.float64, type(old_val)
            assert type(new_val) == float or type(new_val) == np.float64, type(new_val)
            if i in self.problem.categorical_features:
                change_values.append(new_val)
            else:
                change_values.append(new_val - old_val)
            if new_val == old_val:
                changed.append(False)
            elif self.problem.legacy_mode:
                # because of the space transformations there may be precision errors
                if abs(new_val - old_val) < 1e-1:
                    changed.append(False)
                else:
                    changed.append(True)
            else:
                changed.append(True)
            i += 1
        return np.array(change_values), np.array(changed)

    # ! New value is not a change value, but setting it to this value
    # E.g., not x_new = x_old + new_value, but x_new = new_value
    def apply(self, new_value, old_instance):
        assert type(old_instance) == np.ndarray, type(old_instance)
        assert old_instance.dtype == np.float64, old_instance.dtype
        assert type(new_value) == float or type(new_value) == np.float64, type(
            new_value
        )

        new_instance = self.tweak(new_value, old_instance)

        assert new_instance.dtype == np.float64, new_instance.dtype
        assert type(new_instance) == np.ndarray
        assert len(new_instance) == len(old_instance)

        change_values, changed = self.get_changed_values(old_instance, new_instance)
        assert len(change_values) == len(old_instance)

        penalty = self.get_penalty(old_instance, new_instance, change_values)
        assert type(penalty) == float or type(penalty) == np.float64, type(penalty)
        assert penalty >= 0.0, penalty

        if not any(changed) or penalty > 0.0:
            penalty = 1.0
            return infty, infty, penalty, old_instance

        # applying an action must always have an effect!
        assert new_instance[self.feature_idx] != old_instance[self.feature_idx], (
            f"Applying action {self.action_name} did not do anything. Tweak value was {new_value} for feature index {self.feature_idx}",
            old_instance,
            new_instance,
            changed,
        )

        assert type(new_instance) == type(old_instance), (
            type(new_instance),
            type(old_instance),
        )

        # is a scalar
        costs = self.get_costs(old_instance, new_instance)
        assert type(costs) == float or type(costs) == np.float64, type(costs)

        # is a scalar, id added in the problem class if used!
        action_discount = self.get_mean_action_discount(old_instance, changed)
        assert (
            type(action_discount) == float or type(action_discount) == np.float64
        ), type(action_discount)

        return costs, action_discount, penalty, new_instance

    def tweak(
        self,
        new_value,
        old_instance,
    ) -> np.ndarray:
        assert type(new_value) == float or type(new_value) == np.float64, type(
            new_value
        )
        return self._tweak(new_value, old_instance.copy())

    def get_penalty(self, old_instance, new_instance, change_values) -> float:
        penalty = self._penalty(old_instance, new_instance, change_values)
        assert penalty >= 0.0
        assert type(penalty) == float or type(penalty) == np.float64, type(penalty)
        return penalty

    def get_costs(self, old_state, new_state) -> float:
        costs = self._get_costs(old_state, new_state)
        assert costs >= 0.0
        assert type(costs) == float or type(costs) == np.float64, type(costs)
        return max(0.0, costs)

    def get_mean_action_discount(self, old_state, affected_features):
        _affected_features = np.arange(len(old_state))[affected_features]
        mean_action_discount = self.dependency_graph.get_mean_action_discount(
            old_state, _affected_features
        )
        assert (
            type(mean_action_discount) == float
            or type(mean_action_discount) == np.float64
        ), type(mean_action_discount)
        assert (
            mean_action_discount is not None and mean_action_discount is not np.nan
        ), mean_action_discount
        assert 0.0 <= mean_action_discount <= 1.0, mean_action_discount
        return mean_action_discount

    def _get_costs(self, old_state, new_state) -> float:
        """ How to compute costs for this action"""
        raise Exception("Function not defined")

    def _tweak(self, new_value, old_instance) -> np.ndarray:
        """ How to tweak for this action"""
        raise Exception("Function not defined")

    def _penalty(self, old_instance, new_instance, change_values) -> float:
        """ How to compute penalty for this action"""
        raise Exception("Function not defined")
