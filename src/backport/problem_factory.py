import copy

import numpy as np
import pandas as pd

from pymoo.model.problem import Problem


from sequential.sequence import Sequence
from cscf.decoder import Decoder
from util.gowers import gower_matrix

from backport.wrapper.actions_wrapper import ActionWrapper

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from feature_cost_model.feature_dependency_graph import FeatureDependencyGraph


class ProblemFactory(Problem):
    def __init__(
        self,
        dataset,
        actions: dict,
        original_instance,
        target_class,
        blackbox_classifier,
        blackbox_probabilitiy,
        # * form: NUM: action_idx: [lower, upper]
        # * form: CAT: action_idx: [values]
        bounds_and_values: dict,
        feature: dict,
        categorical_features: list,
        dependency_graph,
        chosen_objectives=[
            "feature_tweaking_frequencies",
            "summed_costs",
            "gowers_distance",
        ],
        legacy_mode=False,
        decoder=None,
    ):

        self.legacy_mode = legacy_mode
        self.x0 = original_instance.copy()
        assert type(self.x0) == np.ndarray, type(self.x0)
        assert self.x0.ndim == 1, self.x0.ndim
        assert self.x0.dtype == np.float64, self.x0.dtype
        self.target_class = target_class
        self.available_actions = actions
        self.n_actions = len(actions)
        self.max_sequence_length = self.n_actions

        self.categorical_features = categorical_features

        self.feature = feature

        self.dependency_graph = dependency_graph
        if self.dependency_graph is None:
            self.dependency_graph = FeatureDependencyGraph(None)

        self.chosen_objectives = chosen_objectives

        # used only for checking is values changed etc.
        self.dummy_actions = {
            key: self.get_wrapped_action(params) for key, params in actions.items()
        }

        self.bounds_and_values = bounds_and_values

        self.dataset = dataset

        self.blackbox = blackbox_classifier
        self.prob_blackbox = blackbox_probabilitiy

        # Create idx mapping of action types
        self.real_actions_idx = [
            key
            for key, action in self.available_actions.items()
            if not action["is_categorical"]
        ]
        self.cat_actions_idx = [
            key
            for key, action in self.available_actions.items()
            if action["is_categorical"]
        ]

        # Prepate allowed categorical values mapping
        self.xxl = np.zeros(self.n_actions, dtype=np.float64)
        self.xxu = np.zeros(self.n_actions, dtype=np.float64)
        assert self.n_actions == len(self.bounds_and_values)
        for dict_id, action in self.available_actions.items():
            provided_values_or_range = self.bounds_and_values[dict_id]
            if action["is_categorical"]:
                # Create a mapping of #upper values
                upper = len(provided_values_or_range)
                # lower bound can stay, upper adjusts
                lower = 0.0
                self.xxu[dict_id] = upper
            else:
                lower, upper = provided_values_or_range
                self.xxl[dict_id] = lower
                self.xxu[dict_id] = upper
            assert lower < upper, (provided_values_or_range, lower, upper)
        # assert there is a range
        assert len(np.where(self.xxl == self.xxu)[0]) == 0.0, (self.xxl, self.xxu)

        self.num_ranges = np.zeros(len(self.dataset.real_features), dtype=np.float64)
        self.num_max = np.zeros(len(self.dataset.real_features), dtype=np.float64)
        if self.legacy_mode:
            dat = self.dataset.xs[:, :-1]
        else:
            dat = self.dataset.get_numpy_representation()
        for i, col in enumerate(self.dataset.real_features):
            col_array = dat[:, col].astype(np.float32)
            _max = np.nanmax(col_array)
            _min = np.nanmin(col_array)

            if np.isnan(_max):
                _max = 0.0
            if np.isnan(_min):
                _min = 0.0
            self.num_max[i] = _max
            self.num_ranges[i] = (1 - _min / _max) if (_max != 0) else 0.0
        assert self.num_ranges.dtype == np.float64, self.num_ranges.dtype
        assert self.num_max.dtype == np.float64, self.num_max.dtype

        self.invalid_costs = np.inf

        self.n_unique_actions = len(set([a["action_id"] for key, a in actions.items()]))

        # Each action has inidivudal costs in our formulation
        # Additionally, we add gowers_distance
        # and the sequence length
        n_objectives = 0
        if "summed_costs" in self.chosen_objectives:
            n_objectives += 1
        if "feature_tweaking_frequencies" in self.chosen_objectives:
            n_objectives += len(self.x0)
        if "summed_costs_discounted" in self.chosen_objectives:
            n_objectives += 1
        if "gowers_distance" in self.chosen_objectives:
            n_objectives += 1

        n_constraints = 3

        super().__init__(
            # Twice since we have the sequence and numerical part of equal lengths
            n_var=self.n_actions * 2,
            n_obj=n_objectives,
            n_constr=n_constraints,
            elementwise_evaluation=False,
            # General algorithm bounds are [0,1] for each variable since
            # we use BRKGA. The decoder will translate it for us to the actual space.
            # xl=np.zeros(self.max_sequence_length),
            # xu=np.ones(self.max_sequence_length),
            xl=np.zeros(self.n_actions * 2),
            xu=np.ones(self.n_actions * 2),
        )

        # For translation of the bounds
        if decoder is None:
            self.decoder = Decoder(self)
        else:
            self.decoder = decoder(self)

    def _evaluate(self, x, out, *args, **kwargs):
        assert self.x0.dtype == np.float64
        assert x.dtype == np.float64, x.dtype
        # Use the decoder to map each value to the actual range
        assert x.ndim == 2, x.ndim
        # decoded_x = x
        decoded_x = np.array(
            [self.decoder.decode(instance) for instance in x], dtype=np.float64
        )
        # print(pd.DataFrame(decoded_x[:,6:]))
        assert len(decoded_x) == len(x)
        assert decoded_x.dtype == np.float64

        # Compute sequences and tweaking_values and check if they are valid
        valid_seqs = np.array([self.check_valid_seq(sol) for sol in decoded_x])
        assert len(valid_seqs) == len(decoded_x)

        sequences = [
            self.create_sequence(sol) if valid_seqs[i] else None
            for i, sol in enumerate(decoded_x)
        ]
        assert len(sequences) == len(decoded_x)

        tweaking_values = [
            self.get_tweaking_values(sol) if valid_seqs[i] else None
            for i, sol in enumerate(decoded_x)
        ]
        assert len(tweaking_values) == len(decoded_x)

        # Compute objectives
        seq_lengths = [
            self.seq_length(sol) if valid_seqs[i] else self.invalid_costs
            for i, sol in enumerate(decoded_x)
        ]
        assert len(seq_lengths) == len(decoded_x)

        (
            tweaked_instances,
            costs,
            discounts,
            penalties,
        ) = self.get_individual_costs_and_tweaked_instance(tweaking_values, sequences)

        assert costs.shape == discounts.shape

        assert costs.shape[1] == self.n_actions, costs.shape[1]

        # ! round costs to keep sanity
        # ! Does not really interfere with the comparison (4th digit after comma)
        decimal_places = 4
        costs = np.around(costs, decimal_places)
        assert costs.dtype == np.float64, costs.dtype

        assert len(costs) == len(decoded_x)
        assert len(tweaked_instances) == len(decoded_x)
        assert len(penalties) == len(decoded_x)

        # build fitness vector
        fitness_vec = []
        if "summed_costs" in self.chosen_objectives:
            fitness_vec.append(costs.sum(axis=1))
        if "summed_costs_discounted" in self.chosen_objectives:
            dc = costs * discounts
            assert dc.shape == costs.shape, (dc.shape, costs.shape)
            fitness_vec.append(dc.sum(axis=1))
        if "feature_tweaking_frequencies" in self.chosen_objectives:
            fitness_vec.append(
                self.get_feature_tweaking_frequencies(tweaking_values, sequences)
            )
        if "gowers_distance" in self.chosen_objectives:
            distances = self.get_gowers_distance(tweaked_instances, valid_seqs)
            fitness_vec.append(distances)

        fitness = np.column_stack(fitness_vec)
        assert fitness.dtype == np.float64, fitness.dtype

        g1 = self.check_target_class_condition(tweaked_instances, valid_seqs)
        g2 = [1.0 if not v else 0.0 for v in valid_seqs]
        g3 = penalties

        penalty = np.column_stack([g1, g2, g3])
        assert penalty.dtype == np.float64, penalty.dtype

        assert len(penalty) == len(fitness)
        assert len(fitness) == len(decoded_x)

        assert fitness.shape[1] == self.n_obj, fitness.shape[1]
        assert penalty.shape[1] == self.n_constr, penalty.shape[1]

        out["F"] = fitness.astype(float)
        out["G"] = penalty
        out["pheno"] = decoded_x
        hashs = np.array(
            [
                hash(
                    str(
                        sorted(
                            [
                                self.available_actions[i]["action_id"]
                                for i in dx[: self.max_sequence_length]
                                if i != -1
                            ]
                        )
                    )
                )
                for dx in decoded_x
            ]
        )
        assert len(hashs) == len(decoded_x)
        out["hash"] = hashs
        out["tweaked_instances"] = tweaked_instances
        out["action_costs"] = costs
        out["action_discounts"] = discounts
        # always additionally log these two values
        out["summed_costs"] = costs.sum(axis=1)
        out["summed_costs_discounted"] = (costs * discounts).sum(axis=1)

    def get_individual_costs_and_tweaked_instance(self, tweaking_values, sequences):
        n_sols = len(tweaking_values)
        costs = np.zeros((n_sols, self.n_actions), dtype=float)
        discounts = np.ones((n_sols, self.n_actions), dtype=float)
        penalties = np.zeros(n_sols, dtype=float)
        tweaked_instances = []
        for i, tweak in enumerate(tweaking_values):
            if tweak is None:
                tweaked_instances.append(np.full(len(self.x0), np.nan))
                costs[i, :] = np.full(self.n_actions, self.invalid_costs)
            else:
                (
                    tweaked_instance,
                    sequence_costs,
                    sequence_discounts,
                    penalty,
                ) = sequences[i].unroll_actions_individual_costs(
                    self.x0.copy(),
                    tweak,
                    self.n_actions,
                )
                assert penalty.ndim == 1
                assert sequence_costs.ndim == 1
                assert len(tweaked_instance) == len(self.x0)
                tweaked_instances.append(tweaked_instance)
                costs[i, :] = sequence_costs
                discounts[i, :] = sequence_discounts
                penalties[i] += penalty.sum()
        return (
            np.array(tweaked_instances),
            costs,
            discounts,
            penalties,
        )

    def check_valid_seq(self, x):
        assert x.ndim == 1, x.ndim
        split_point = self.max_sequence_length
        action_order_part = x[:split_point]
        if all(action_order_part == -1):
            return False
        return True

    def check_target_class_condition(self, x, valid):
        _x = x[valid]
        preds = self.blackbox(np.array(_x))
        output = np.full(len(x), 1.0)
        correct = np.ones(len(_x))
        correct[preds == self.target_class] = 0.0
        output[valid] = correct
        return output

    def get_feature_tweaking_frequencies(self, tweaking_values, sequences):
        n_sols = len(tweaking_values)
        res = np.zeros((n_sols, len(self.x0)), dtype=float)
        for i, tweak in enumerate(tweaking_values):
            if tweak is None:
                res[i, :] = np.full(len(self.x0), self.invalid_costs)
            else:
                tweaked_instances = sequences[i].get_tweaked_instance_after_each_action(
                    self.x0.copy(),
                    tweak,
                )
                assert len(tweaked_instances) == len(tweak)
                old_instance = self.x0.copy()
                for new_instance in tweaked_instances:
                    res[i] += self.get_feature_changes(old_instance, new_instance)
                    old_instance = new_instance
        return res

    def get_feature_changes(self, old_instance, new_instance):
        assert old_instance.ndim == 1
        assert new_instance.ndim == 1
        assert len(old_instance) == len(new_instance)
        res = np.zeros(len(old_instance))
        diffs = new_instance != old_instance
        res[diffs] = 1.0
        return res

    def create_sequence(self, x, prin=False) -> Sequence:
        assert x.ndim == 1, x.ndim
        split_point = self.max_sequence_length
        action_order_part = x[:split_point]
        active = action_order_part[action_order_part != -1]

        actions_params = [copy.copy(self.available_actions[i]) for i in active]
        legacy_actions = [a["legacy_action"] for a in actions_params]

        if self.legacy_mode:
            num_params = sum([action.num_params for action in legacy_actions], 0)
            i = 0
            for cur_idx, legacy_action in enumerate(legacy_actions):
                actions_params[cur_idx]["legacy_action"] = copy.copy(
                    legacy_action
                ).set_p_selector(i, num_params, add_tensor=False)
                i += legacy_action.num_params

        actions = [self.get_wrapped_action(params) for params in actions_params]
        seq = Sequence(actions, round_costs=False, print_application=prin)
        return seq

    def get_wrapped_action(self, params):
        if self.legacy_mode:
            return ActionWrapper(
                legacy_action=params["legacy_action"],
                action_id=params["action_id"],
                initial_instance=self.x0,
                feature_idx=params["feature_idx"],
                problem=self,
                translator=params["translator"],
                is_categorical=params["is_categorical"],
                feature_idx_in_z=params["feature_idx_in_z"],
                feat_name=params["feat_name"],
                dependency_graph=self.dependency_graph,
                key=params["key"],
            )
        else:
            action = params["legacy_action"](
                action_id=params["action_id"],
                initial_instance=self.x0,
                feature_idx=params["feature_idx"],
                problem=self,
                dependency_graph=self.dependency_graph,
                key=params["key"],
            )
            return action

    def get_tweaking_values(self, x) -> np.array:
        assert x.ndim == 1, x.ndim
        split_point = self.max_sequence_length
        action_order_part = x[:split_point].astype(int)
        value_part = x[split_point:]
        idxs = action_order_part[action_order_part != -1]
        tweaking_values = value_part[idxs].copy()
        return tweaking_values

    def seq_length(self, x) -> int:
        assert x.ndim == 1, x.ndim
        split_point = self.max_sequence_length
        action_order_part = x[:split_point]
        return len(action_order_part[action_order_part != -1])

    def get_gowers_distance(self, x, valid) -> float:
        _x = x.copy()[valid]
        original_cat_idx = self.dataset.cat_features

        gowers = gower_matrix(
            _x,
            self.x0.reshape(1, -1),
            cat_features=original_cat_idx,
            norm=self.num_max,
            norm_ranges=self.num_ranges,
        )
        assert gowers.dtype == np.float64 or gowers.dtype == np.float32, gowers.dtype
        assert len(gowers.flatten()) == len(_x)
        gowers = gowers.flatten()
        # round to avoid being too sensitive
        gowers = np.around(gowers, 2)
        assert gowers.ndim == 1, gowers.shape
        res = np.full(len(x), 1.0, dtype=np.float64)
        res[valid] = gowers
        return res
