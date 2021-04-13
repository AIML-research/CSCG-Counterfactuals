class Constraint:
    def __init__(self, feature_idx, name=None):
        self.name = name
        self.feature_idx = feature_idx
        self.former_constraint = None

    def AND(self, constraint):
        self.former_constraint = constraint
        return self

    def validate(self, old_instance, new_instance):
        if self.former_constraint is not None:
            return self._validate(
                old_instance, new_instance
            ) & self.former_constraint.validate(old_instance, new_instance)
        return self._validate(old_instance, new_instance)

    def _validate(self, old_instance, new_instance):
        raise Exception("Not implemented")


class BoundaryConstraint(Constraint):
    def __init__(self, lower_bound, upper_bound, *args, **kwargs):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # in case initial value is already outside bounds
        if self.lower_bound > self.upper_bound:
            self.upper_bound = self.lower_bound
        assert self.lower_bound <= self.upper_bound
        super().__init__(*args, **kwargs)

    def _validate(self, old_instance, new_instance):
        return (new_instance[self.feature_idx] >= self.lower_bound) & (
            new_instance[self.feature_idx] <= self.upper_bound
        )


class InValueSetConstraint(Constraint):
    def __init__(self, values, *args, **kwargs):
        self.values = values
        assert len(self.values) > 0
        super().__init__(*args, **kwargs)

    def _validate(self, old_instance, new_instance):
        return new_instance[self.feature_idx] in self.values


class OnlyIncreaseConstraint(Constraint):
    def __init__(self, initial_value, *args, **kwargs):
        self.initial_value = initial_value
        super().__init__(*args, **kwargs)

    def _validate(self, old_instance, new_instance):
        change = new_instance[self.feature_idx] - old_instance[self.feature_idx]
        return (change > 0.0) & (new_instance[self.feature_idx] > self.initial_value)


class PreRequirementConstraint(Constraint):
    def __init__(self, pre_condition_value, *args, **kwargs):
        self.pre_condition_value = pre_condition_value
        super().__init__(*args, **kwargs)

    def _validate(self, old_instance, new_instance):
        return old_instance[self.feature_idx] == self.pre_condition_value


class PostRequirementConstraint(Constraint):
    def __init__(self, post_condition_value, *args, **kwargs):
        self.post_condition_value = post_condition_value
        super().__init__(*args, **kwargs)

    def _validate(self, old_instance, new_instance):
        return new_instance[self.feature_idx] == self.post_condition_value
