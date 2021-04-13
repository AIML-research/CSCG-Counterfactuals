import numpy as np

from sequential.action import Action
from backport.utils import RepresentationTranslator


class ActionWrapper(Action):
    """
    This class wraps the competitor methods action and cost model.
    """

    def __init__(
        self,
        action_id,
        legacy_action,
        translator,  # RepresentationTranslator(feature_objects)
        feature_idx,
        feature_idx_in_z,
        initial_instance,
        is_categorical,
        problem,
        feat_name,
        dependency_graph=None,
        *args,
        **kwargs,
    ):
        self.legacy_action = legacy_action

        self.translator = translator  # RepresentationTranslator(feature_objects)

        self.initial_instance_in_z = self.translator.instance_to_z(
            initial_instance.copy()
        )
        self.feat_name = feat_name
        self.feature_idx_in_z = feature_idx_in_z

        super().__init__(
            action_id=action_id,
            feature_idx=feature_idx,
            initial_instance=initial_instance,
            action_name=self.legacy_action.name,
            is_categorical=is_categorical,
            problem=problem,
            description=self.legacy_action.description,
            feature_name=self.legacy_action.target_features,
            round_costs=False,
            print_application=False,
            dependency_graph=dependency_graph,
            *args,
            **kwargs,
        )

    def _get_costs(self, old_state, new_state):
        old_instance_in_z_repr = self.translator.instance_to_z(old_state)
        new_instance_in_z_repr = self.translator.instance_to_z(new_state)

        # costs = self.legacy_action.get_cost(
        #     self.initial_instance_in_z, new_instance_in_z_repr, use_tensor=False
        # )
        # ! Changed to this since we need to consider the prior instance in the SEQUENCE
        # ! this doesn't have to be x0
        costs = self.legacy_action.get_cost(
            old_instance_in_z_repr, new_instance_in_z_repr, use_tensor=False
        )
        if costs == np.inf:
            # If invalid action tweak, return high costs
            # Rest is handled by penalty
            return 1e20
        return costs

    def _tweak(self, new_value, old_instance) -> np.ndarray:
        original_instance_in_z = self.translator.instance_to_z(old_instance)

        tweak_value = new_value
        new_instance = self.legacy_action.apply(
            original_instance_in_z, tweak_value, use_tensor=False
        )

        new_instance = self.translator.instance_to_x(new_instance)
        assert new_instance.dtype == old_instance.dtype, (
            new_instance.dtype,
            old_instance.dtype,
        )
        # ! round here as otherwise very small changes have an impact
        new_instance = np.around(new_instance, 3)
        return new_instance

    def _penalty(self, old_instance, new_instance, change_values):
        old_instance_in_z_repr = self.translator.instance_to_z(old_instance)
        new_instance_in_z_repr = self.translator.instance_to_z(new_instance)

        costs = self.legacy_action.get_cost(
            old_instance_in_z_repr, new_instance_in_z_repr, use_tensor=False
        )
        if costs == np.inf:
            return 1.0
        return 0.0
