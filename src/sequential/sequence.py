"""

General remark: WE ASSUME AN INSTANCE CAN ONLY BE CHANGED/TWEAKED THROUGH AVAILABLE ACTIONS! 

"""
# import copy
import numpy as np
import pandas as pd
import copy


class Sequence:
    """
    Creates a sequence of actions object of which certain tasks can be applied.
    E.g. applying all actions sequentially.
    """

    def __init__(self, actions: list, round_costs=False, print_application=False):
        self.length = len(actions)
        self.sequence = actions
        self.round_costs = round_costs
        if self.round_costs:
            for a in self.sequence:
                a.round_costs = True

        self.log = print_application
        if self.log:
            for a in self.sequence:
                a.log = True

    def __str__(self):
        return "Sequence(" + " -> ".join([a.action_name for a in self.sequence]) + ")"

    def unroll_actions(self, instance, tweaking_values):
        """
        Applies all actions in the sequence one after another.
        This tweaks the running instance sequentially and updates all associated costs for each action

        Parameters
        ----------
        instance : np.array
            The original instance on which the action sequence shall be applied
        tweaking_values : list or np.array
            The tweaking values for each action that will be applied by taking the respective action

        Returns
        -------
        [type]
            The final tweaked version of 'instance' as well as the total global and local costs after all actions
        """
        sequence_costs = 0.0
        penalties = 0.0
        tweaked_instance = instance.copy()
        for i, action in enumerate(self.sequence):
            costs, discount, penalty, tweaked_instance = action.apply(
                tweaking_values[i], tweaked_instance
            )
            sequence_costs += costs
            penalties += penalty

        if self.log:
            print(
                "Costs after applying the sequence with tweaks",
                tweaking_values,
                "is",
                sequence_costs,
            )
            print("Penalty is", penalties)
        return (tweaked_instance, sequence_costs, penalties)

    def unroll_actions_individual_costs(
        self, instance, tweaking_values, max_action_length
    ):
        """
        Returns the costs per action, instead of a sum.
        The length of the returned cost array is equal to the max number of possible actions.
        The cost array contains zeros, when an action was not used

        Parameters
        ----------
        instance : [type]
            [description]
        tweaking_values : [type]
            [description]
        max_action_length : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        sequence_costs = np.zeros(max_action_length)
        sequence_discounts = np.ones(max_action_length)
        penalties = []
        tweaked_instance = instance.copy()
        for i, action in enumerate(self.sequence):
            (
                costs,
                discount,
                penalty,
                tweaked_instance,
            ) = action.apply(tweaking_values[i], tweaked_instance)
            action_idx = action.key
            # not set before
            assert sequence_costs[action_idx] == 0.0, sequence_costs[action_idx]
            assert sequence_discounts[action_idx] == 1.0, sequence_discounts[action_idx]

            sequence_costs[action_idx] = costs
            sequence_discounts[action_idx] = discount
            penalties.append(penalty)
        penalties = np.array(penalties)
        # if there was some problem with the discount functions, then check
        # if this is also reflected in the penalties (discount should otherwise never be greater than 1.0)
        if (sequence_discounts > 1.0).any():
            assert (penalties > 0.0).any(), (sequence_discounts, penalties)

        return (tweaked_instance, sequence_costs, sequence_discounts, penalties)

    def get_tweaked_instance_after_each_action(self, instance, tweaking_values):
        """
        Applies each action in sequence onto the given instance and outputs the tweaked version after each step.

        Parameters
        ----------
        instance : [type]
            The original instance that shall be tweaked
        tweaking_values : [type]
            The tweaking values for each action

        Returns
        -------
        np.array of length #actions
            Tweaked instances after each sequential action
        """
        tweaked_instance = instance.copy()
        tweaked_instances = []
        for i, action in enumerate(self.sequence):
            tweaked_instance = action.tweak(tweaking_values[i], tweaked_instance)
            tweaked_instances.append(tweaked_instance.copy())
        return np.array(tweaked_instances, dtype=np.float64)
