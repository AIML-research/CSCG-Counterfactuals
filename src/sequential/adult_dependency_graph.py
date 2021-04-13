# dependency graph
import numpy as np
from feature_cost_model.feature_dependency_graph import FeatureDependencyGraph


def get_adult_dependency_graph(features):
    def edu_capital(x):
        min_discount = 1.0
        max_discount = 0.5
        # thresh for this value in the competitor model is 16.5 and we add 1 to include the 16.5
        n_levels = 16 + 1
        discounts = np.linspace(max_discount, min_discount, num=n_levels)[::-1]
        edu_level = int(x[features["Education num"]])
        return discounts[edu_level]

    def edu_occupation(x):
        min_discount = 1.0
        max_discount = 0.5
        # thresh for this value in the competitor model is 16.5 and we add 1 to include the 16.5
        n_levels = 16 + 1
        discounts = np.linspace(max_discount, min_discount, num=n_levels)[::-1]
        edu_level = int(x[features["Education num"]])
        return discounts[edu_level]

    def edu_workclass(x):
        min_discount = 1.0
        max_discount = 0.5
        # thresh for this value in the competitor model is 16.5 and we add 1 to include the 16.5
        n_levels = 16 + 1
        discounts = np.linspace(max_discount, min_discount, num=n_levels)[::-1]
        edu_level = int(x[features["Education num"]])
        return discounts[edu_level]

    def workhrs_capital(x):
        work_hrs = x[features["Hours/Week"]]
        mi = np.exp(0)
        ma = np.exp(80)
        discount = np.interp(np.exp(work_hrs), [mi, ma], [0.0, 1.0])
        return 1.0 - discount

    def workhrs_edu(x):
        work_hrs = x[features["Hours/Week"]]
        if work_hrs < 20:
            return 0.5
        elif work_hrs < 30:
            return 0.6
        elif work_hrs <= 40:
            return 0.7
        else:
            return 1.0

    G = {
        (features["Education num"], features["Capital Gain"]): edu_capital,
        (features["Education num"], features["Occupation"]): edu_occupation,
        (features["Education num"], features["Workclass"]): edu_workclass,
        (features["Hours/Week"], features["Capital Gain"]): workhrs_capital,
        (features["Hours/Week"], features["Education num"]): workhrs_edu,
    }
    G = FeatureDependencyGraph(G, False)
    return G
