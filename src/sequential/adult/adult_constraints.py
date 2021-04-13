from ..constraints import (
    Constraint,
    BoundaryConstraint,
    InValueSetConstraint,
    OnlyIncreaseConstraint,
    PostRequirementConstraint,
    PreRequirementConstraint,
)


class AgeOnlyIncreaseAndInBounds(Constraint):
    def __init__(self, initial_value, max_value, feature_idx):
        self.initial_value = initial_value
        super().__init__(feature_idx=feature_idx)
        self.increase_only = OnlyIncreaseConstraint(
            initial_value=self.initial_value, feature_idx=feature_idx
        )
        self.in_bounds = BoundaryConstraint(
            lower_bound=self.initial_value,
            upper_bound=max_value,
            feature_idx=feature_idx,
        )

    def _validate(self, *args, **kwargs):
        return self.increase_only.AND(self.in_bounds).validate(*args, **kwargs)


class EducationOnlyIncreaseAndInBounds(Constraint):
    def __init__(self, initial_value, allowed_values, feature_idx):
        self.initial_value = initial_value
        super().__init__(feature_idx=feature_idx)
        self.increase_only = OnlyIncreaseConstraint(
            initial_value=self.initial_value, feature_idx=feature_idx
        )
        self.in_bounds = InValueSetConstraint(
            feature_idx=feature_idx, values=allowed_values
        )

    def _validate(self, *args, **kwargs):
        return self.increase_only.AND(self.in_bounds).validate(*args, **kwargs)


class WorkHrsInBounds(BoundaryConstraint):
    def __init__(self, lower_bound, upper_bound, feature_idx):
        super().__init__(lower_bound, upper_bound, feature_idx=feature_idx)


class CapitalInBounds(BoundaryConstraint):
    def __init__(self, lower_bound, upper_bound, feature_idx):
        super().__init__(lower_bound, upper_bound, feature_idx=feature_idx)


class OccupationInBounds(InValueSetConstraint):
    def __init__(self, values, feature_idx):
        super().__init__(values, feature_idx=feature_idx)


class WorkclassInBounds(InValueSetConstraint):
    def __init__(self, values, feature_idx):
        super().__init__(values, feature_idx=feature_idx)
