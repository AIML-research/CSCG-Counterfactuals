import numpy as np

from competitor.actions.feature import CategoricFeature, Feature


class RepresentationTranslator:
    def __init__(self, features_objects):
        self.features = features_objects
        self.idx_to_feat_map = {
            feat.orig_idx: feat.name for feat in self.features.values()
        }
        self.feat_to_idx_map = {feat: idx for idx, feat in self.idx_to_feat_map.items()}
        self.n_features = len(features_objects)
        self.n_features_z = sum(
            [
                len(feat.values) if feat.iscat else 1
                for feat in features_objects.values()
            ]
        )

        self.add_mean = True

    def instance_to_x(self, instance_in_z):
        transformed_instance = np.zeros(self.n_features)
        for _, feature in self.features.items():
            i = feature.orig_idx
            if not feature.iscat:
                to_x = feature.ztox(instance_in_z[feature.idx], add_mean=self.add_mean)
            elif feature.iscat:
                # get index
                to_x = feature.get_feature_value(instance_in_z, use_tensor=False)
                # get original value
                to_x = feature.values[to_x]
            transformed_instance[i] = to_x
        return transformed_instance

    def instance_to_z(self, instance_in_x):
        transformed_instance = np.zeros(self.n_features_z)
        i = 0
        for j, value in enumerate(instance_in_x):
            name = self.idx_to_feat_map[j]
            feature = self.features[name]
            if not feature.iscat:
                to_z = feature.xtoz(value, add_mean=self.add_mean)
                transformed_instance[i] = to_z
                i += 1
            elif feature.iscat:
                idx = feature.values_dict[int(value)]
                transformed_instance[i + idx] = 1.0
                i += len(feature.values)

        return transformed_instance
