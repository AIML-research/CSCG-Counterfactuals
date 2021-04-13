import numpy as np

from ..action import Action
from sequential.adult.adult_constraints import (
    AgeOnlyIncreaseAndInBounds,
    EducationOnlyIncreaseAndInBounds,
    WorkclassInBounds,
    WorkHrsInBounds,
    OccupationInBounds,
    CapitalInBounds,
)
from sequential.adult.adult_costs import (
    IncreaseAgeCosts,
    IncreaseEducationCosts,
    ChangeWorkHrsCosts,
    ChangeCategoricalCosts,
    IncreaseCapitalGainCosts,
)


class IncreaseAge(Action):
    def __init__(
        self,
        action_id: int,
        initial_instance,
        feature_idx: int,
        problem,
        dependency_graph=None,
        *args,
        **kwargs,
    ):

        super().__init__(
            action_id=action_id,
            feature_idx=feature_idx,
            initial_instance=initial_instance.copy(),
            problem=problem,
            action_name="Wait X Years",
            is_categorical=False,
            description="Increases the feature age by the respective value.",
            feature_name="Age",
            dependency_graph=dependency_graph,
            *args,
            **kwargs,
        )

        self.costs = IncreaseAgeCosts(problem.feature, dependency_graph)

    def _get_costs(self, old_state, new_state):
        return self.costs.get_costs(old_state, new_state)

    def _tweak(self, new_value, old_instance) -> np.ndarray:
        new_instance = old_instance.copy()
        new_instance[self.feature_idx] = new_value
        return new_instance

    def _penalty(self, old_instance, new_instance, change_values) -> float:
        tweak_value = change_values[self.feature_idx]
        penalty = max(1.0, abs(tweak_value))

        constraint = AgeOnlyIncreaseAndInBounds(
            initial_value=self.initial_value,
            max_value=68,  # we assume below is retirement
            feature_idx=self.feature_idx,
        )

        if not constraint.validate(old_instance, new_instance):
            return penalty
        return 0.0


class ChangeEducation(Action):
    def __init__(
        self,
        action_id: int,
        initial_instance,
        feature_idx: int,
        problem,
        dependency_graph=None,
        *args,
        **kwargs,
    ):

        super().__init__(
            action_id=action_id,
            feature_idx=feature_idx,
            initial_instance=initial_instance.copy(),
            problem=problem,
            action_name="Change Education",
            is_categorical=True,
            description="Changes the education level with respect to the intial one. I.e. education can only increase.",
            feature_name="Education",
            dependency_graph=dependency_graph,
            *args,
            **kwargs,
        )

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

        self.education_costs = IncreaseEducationCosts(problem.feature, dependency_graph)
        self.age_costs = IncreaseAgeCosts(problem.feature, dependency_graph)

    def _get_costs(self, old_state, new_state):
        return self.education_costs.get_costs(
            old_state, new_state
        ) + self.age_costs.get_costs(old_state, new_state)

    def _tweak(self, new_value, old_instance) -> np.ndarray:
        new_instance = old_instance.copy()

        # update education level
        new_instance[self.feature_idx] = new_value

        if new_instance[self.feature_idx] == old_instance[self.feature_idx]:
            return old_instance
        # applying an action must always have an effect!
        # * This is already checked in the .tweak meta method
        assert new_instance[self.feature_idx] != old_instance[self.feature_idx]

        # update age
        current_level = self.education_level_order.index(old_instance[self.feature_idx])
        age_increase = (
            sum(self.general_costs[current_level + 1 : int(new_value) + 1])
        ) + 1.0
        work_hrs = np.interp(
            old_instance[self.problem.feature["hours_per_week"]], [0, 30], [0.5, 1]
        )
        new_instance[self.problem.feature["age"]] += age_increase * work_hrs
        # applying an action must always have an effect!
        assert (
            new_instance[self.problem.feature["age"]]
            != old_instance[self.problem.feature["age"]]
        )

        # # ! Testing
        # # when increasing degree, you lose your job
        # new_instance[self.problem.feature["workclass"]] = 0.0
        # # assert (
        # #     new_instance[self.problem.feature["workclass"]]
        # #     != old_instance[self.problem.feature["workclass"]]
        # # )
        # new_instance[self.problem.feature["occupation"]] = 0.0
        # # assert (
        # #     new_instance[self.problem.feature["occupation"]]
        # #     != old_instance[self.problem.feature["occupation"]]
        # # )
        # new_instance[self.problem.feature["hours_per_week"]] = 0.0

        return new_instance

    def _penalty(self, old_instance, new_instance, change_values) -> float:
        former_level = self.education_level_order.index(old_instance[self.feature_idx])

        penalty = 1.0

        constraint = EducationOnlyIncreaseAndInBounds(
            initial_value=self.initial_value,
            allowed_values=self.education_level_order[former_level:],
            feature_idx=self.feature_idx,
        ).AND(
            AgeOnlyIncreaseAndInBounds(
                initial_value=old_instance[self.problem.feature["age"]],
                max_value=68,  # we assume below is retirement,
                feature_idx=self.problem.feature["age"],
            )
        )
        if not constraint.validate(old_instance, new_instance):
            return penalty
        return 0.0


class IncreaseCapital(Action):
    def __init__(
        self,
        action_id: int,
        initial_instance,
        feature_idx: int,
        problem,
        dependency_graph=None,
        *args,
        **kwargs,
    ):

        super().__init__(
            action_id=action_id,
            feature_idx=feature_idx,
            initial_instance=initial_instance.copy(),
            problem=problem,
            action_name="Increase Capital",
            is_categorical=False,
            description="Increases the capital gain.",
            feature_name="Capital-gain",
            dependency_graph=dependency_graph,
            *args,
            **kwargs,
        )

        self.capital_gain_costs = IncreaseCapitalGainCosts(
            problem.feature, dependency_graph
        )
        self.age_costs = IncreaseAgeCosts(problem.feature, dependency_graph)

    def _get_costs(self, old_state, new_state):
        return self.capital_gain_costs.get_costs(
            old_state, new_state
        )  # + self.age_costs.get_costs(old_state, new_state)

    def _tweak(self, new_value, old_instance) -> np.ndarray:
        new_instance = old_instance.copy()

        new_instance[self.feature_idx] = new_value
        if new_instance[self.feature_idx] == old_instance[self.feature_idx]:
            return old_instance
        # applying an action must always have an effect
        # * This is already checked in the .tweak meta method
        assert new_instance[self.feature_idx] != old_instance[self.feature_idx]

        # capital_change = abs(new_value - old_instance[self.feature_idx])
        # # 2 years for each 1000$ more
        # age_increase = capital_change / 500
        # new_instance[self.problem.feature["age"]] += age_increase
        # # applying an action must always have an effect!
        # assert (
        #     new_instance[self.problem.feature["age"]]
        #     != old_instance[self.problem.feature["age"]]
        # )
        return new_instance

    def _penalty(self, old_instance, new_instance, change_values) -> float:
        tweak_value = change_values[self.feature_idx]
        penalty = max(1.0, abs(tweak_value))

        constraint = CapitalInBounds(
            lower_bound=0, upper_bound=99999, feature_idx=self.feature_idx
        )
        # ).AND(
        #     AgeOnlyIncreaseAndInBounds(
        #         initial_value=old_instance[self.problem.feature["age"]],
        #         max_value=68,  # we assume below is retirement,
        #         feature_idx=self.problem.feature["age"],
        #     )
        # )

        if not constraint.validate(old_instance, new_instance):
            return penalty
        return 0.0


class ChangeMaritalStatus(Action):
    def __init__(
        self,
        action_id: int,
        initial_instance,
        feature_idx: int,
        problem,
        dependency_graph=None,
        *args,
        **kwargs,
    ):

        super().__init__(
            action_id=action_id,
            feature_idx=feature_idx,
            initial_instance=initial_instance.copy(),
            problem=problem,
            action_name="Change Marital Status",
            is_categorical=True,
            description="Changes the marital status with respect to impossible actions, e.g. divorce without prior marriage.",
            feature_name="Marital-status",
            dependency_graph=dependency_graph,
            *args,
            **kwargs,
        )

        self.costs = ChangeCategoricalCosts(
            self.feature_idx, problem.feature, dependency_graph
        )

    def _get_costs(self, old_state, new_state):
        return self.costs.get_costs(old_state, new_state)

    def _tweak(self, new_value, old_instance) -> np.ndarray:
        new_instance = old_instance.copy()
        new_instance[self.feature_idx] = new_value
        return new_instance

    def _penalty(self, old_instance, new_instance, change_values) -> float:
        new_value = new_instance[self.feature_idx]
        former_value = old_instance[self.feature_idx]

        assume_same = False
        shared_weight = 5.0
        infeasible = 20.0

        # TODO write this as a constraint object
        if former_value == 0:  # is divorced
            if new_value == 1:  # change to married
                return shared_weight if assume_same else 5.0
            elif new_value == 3:  # change to single
                return shared_weight if assume_same else 1.0
            else:
                return infeasible
        elif former_value == 1:  # is married
            if new_value == 0:  # change to divorced
                return shared_weight if assume_same else 5.0
            elif new_value == 2:  # change to separated
                return shared_weight if assume_same else 3.0
            elif new_value == 3:  # change to single
                return shared_weight if assume_same else 5.0
            elif new_value == 4:  # change to widowed
                return shared_weight if assume_same else 10.0
            else:
                return infeasible
        elif former_value == 2:  # is separated
            if new_value == 0:  # change to divorced
                return shared_weight if assume_same else 3.0
            elif new_value == 1:  # change to married
                return shared_weight if assume_same else 3.0
            elif new_value == 3:  # change to single
                return shared_weight if assume_same else 3.0
            elif new_value == 4:  # change to widowed
                return shared_weight if assume_same else 10.0
            else:
                return infeasible
        elif former_value == 3:  # is Single
            if new_value == 1:  # change to married
                return shared_weight if assume_same else 5.0
            else:
                return infeasible
        elif former_value == 4:  # is Widowed
            if new_value == 1:  # change to married
                return shared_weight if assume_same else 8.0
            elif new_value == 3:  # change to single
                return shared_weight if assume_same else 1.0
            else:
                return infeasible
        else:
            return infeasible


class ChangeWorkHours(Action):
    def __init__(
        self,
        action_id: int,
        initial_instance,
        feature_idx: int,
        problem,
        dependency_graph=None,
        *args,
        **kwargs,
    ):

        super().__init__(
            action_id=action_id,
            feature_idx=feature_idx,
            initial_instance=initial_instance.copy(),
            problem=problem,
            action_name="Change Work Hours",
            is_categorical=False,
            description="Change the working hours.",
            feature_name="Working-hours",
            dependency_graph=dependency_graph,
            *args,
            **kwargs,
        )

        self.costs = ChangeWorkHrsCosts(problem.feature, dependency_graph)

    def _get_costs(self, old_state, new_state):
        return self.costs.get_costs(old_state, new_state)

    def _tweak(self, new_value, old_instance) -> np.ndarray:
        new_instance = old_instance.copy()
        new_instance[self.feature_idx] = new_value
        return new_instance

    def _penalty(self, old_instance, new_instance, change_values) -> float:
        tweak_value = change_values[self.feature_idx]
        penalty = max(1.0, abs(tweak_value))

        constraint = WorkHrsInBounds(
            lower_bound=0, upper_bound=80, feature_idx=self.feature_idx
        )

        if not constraint.validate(old_instance, new_instance):
            return penalty
        return 0.0


class ChangeOccupation(Action):
    def __init__(
        self,
        action_id: int,
        initial_instance,
        feature_idx: int,
        problem,
        dependency_graph=None,
        *args,
        **kwargs,
    ):

        super().__init__(
            action_id=action_id,
            feature_idx=feature_idx,
            initial_instance=initial_instance.copy(),
            problem=problem,
            action_name="Change Occupation",
            is_categorical=True,
            description="Changes the occupation.",
            feature_name="Occupation",
            dependency_graph=dependency_graph,
            *args,
            **kwargs,
        )

        self.costs = ChangeCategoricalCosts(
            self.feature_idx, problem.feature, dependency_graph
        )

    def _get_costs(self, old_state, new_state):
        return self.costs.get_costs(old_state, new_state)

    def _tweak(self, new_value, old_instance) -> np.ndarray:
        new_instance = old_instance.copy()
        new_instance[self.feature_idx] = new_value
        return new_instance

    def _penalty(self, old_instance, new_instance, change_values) -> float:
        return 0.0


class ChangeWorkclass(Action):
    def __init__(
        self,
        action_id: int,
        initial_instance,
        feature_idx: int,
        problem,
        dependency_graph=None,
        *args,
        **kwargs,
    ):

        super().__init__(
            action_id=action_id,
            feature_idx=feature_idx,
            initial_instance=initial_instance.copy(),
            problem=problem,
            action_name="Change Workclass",
            is_categorical=True,
            description="Changes the workclass.",
            feature_name="Workclass",
            dependency_graph=dependency_graph,
            *args,
            **kwargs,
        )

        self.costs = ChangeCategoricalCosts(
            self.feature_idx, problem.feature, dependency_graph
        )

    def _get_costs(self, old_state, new_state):
        return self.costs.get_costs(old_state, new_state)

    def _tweak(self, new_value, old_instance) -> np.ndarray:
        new_instance = old_instance.copy()
        new_instance[self.feature_idx] = new_value
        return new_instance

    def _penalty(self, old_instance, new_instance, change_values) -> float:
        return 0.0


class QuitJob(Action):
    def __init__(
        self,
        action_id: int,
        initial_instance,
        feature_idx: int,
        problem,
        dependency_graph=None,
        *args,
        **kwargs,
    ):

        super().__init__(
            action_id=action_id,
            feature_idx=feature_idx,
            initial_instance=initial_instance.copy(),
            problem=problem,
            action_name="Quit Job",
            is_categorical=True,
            description="",
            feature_name="Occupation",
            dependency_graph=dependency_graph,
            *args,
            **kwargs,
        )

        self.costsOccupation = ChangeCategoricalCosts(
            problem.feature["occupation"], problem.feature, dependency_graph
        )
        self.costsWorkclass = ChangeCategoricalCosts(
            problem.feature["workclass"], problem.feature, dependency_graph
        )
        self.costsHrsPerWeek = ChangeWorkHrsCosts(problem.feature, dependency_graph)

    def _get_costs(self, old_state, new_state):
        return (
            self.costsOccupation.get_costs(old_state, new_state)
            + self.costsWorkclass.get_costs(old_state, new_state)
            + self.costsHrsPerWeek.get_costs(old_state, new_state)
        )

    def _tweak(self, new_value, old_instance) -> np.ndarray:
        new_instance = old_instance.copy()
        new_instance[self.problem.feature["workclass"]] = 0.0
        new_instance[self.problem.feature["occupation"]] = 0.0
        new_instance[self.problem.feature["hours_per_week"]] = 0.0
        return new_instance

    def _penalty(self, old_instance, new_instance, change_values) -> float:
        constraint = WorkHrsInBounds(
            lower_bound=0,
            upper_bound=80,
            feature_idx=self.problem.feature["hours_per_week"],
        )

        if not constraint.validate(old_instance, new_instance):
            return 1.0
        return 0.0


actions = {
    0: IncreaseAge,
    1: ChangeEducation,
    2: IncreaseCapital,
    3: ChangeMaritalStatus,
    4: ChangeWorkHours,
    5: ChangeOccupation,
    6: ChangeWorkclass,
    7: QuitJob,
}