import csv
import json
import numpy as np

from competitor.common.paths import GERMAN_DIR


class GermanDataset(object):
    def __init__(self, filename):
        dataset = np.load(filename)
        self.data = dataset[:, :-2]
        self.labels = dataset[:, -2:]

    @classmethod
    def encode_nominal(cls, feature, values):
        encoding = np.zeros(shape=(values.shape[0], feature['num_values']), dtype=np.float32)
        for i in range(values.shape[0]):
            encoding[i][feature['values'].index(values[i])] = 1.
        return encoding

    @classmethod
    def extract_data(cls, filename, feature_spec):
        data = list()
        with open(filename) as file:
            reader = csv.reader(file, delimiter=' ')
            for raw_row in reader:
                row = list()
                for i, feature in enumerate(feature_spec):
                    if feature['type'] == 'numeric':
                        row.append(float(raw_row[i]))
                    elif feature['type'] == 'class':
                        row.append(int(raw_row[i]))
                    else:
                        row.append(int(raw_row[i][1:]))  # Remove prefix 'A'
                data.append(row)
        return np.array(data)

    @classmethod
    def partition(cls, np_data, ratio=0.9):
        train_size = int(np_data.shape[0] * ratio)
        np.random.shuffle(np_data)
        return np_data[:train_size, :], np_data[train_size:, :]

    @classmethod
    def get_feature_stats(cls, xs, feature_spec):
        idx = 0
        feature_stats = list()
        for i, feature in enumerate(feature_spec):
            feature_stat = feature.copy()
            feature_stat['idx'] = idx
            feature_stat['i'] = i
            if feature['type'] == 'numeric':
                feature_stat['num_values'] = 1
                feature_stat['mean'] = np.mean(xs[:, i])
                feature_stat['std'] = np.std(xs[:, i])
                idx += 1
            else:
                feature_stat['num_values'] = len(feature['values'])
                idx += len(feature['values'])
            feature_stats.append(feature_stat)
        return feature_stats

    @classmethod
    def normalize(cls, xs, stats):
        for stat in stats:
            if stat['type'] == 'numeric':
                xs[:, stat['i']] = (xs[:, stat['i']] - stat['mean']) / stat['std']
        return xs

    @classmethod
    def one_hot(cls, data, stats):
        zs = np.ndarray(shape=(data.shape[0], sum([stat['num_values'] for stat in stats])),
                        dtype=np.float32)
        for stat in stats:
            idx = stat['idx']
            num_values = stat['num_values']
            i = stat['i']

            if stat['type'] != 'numeric':
                zs[:, idx:(idx + num_values)] = cls.encode_nominal(stat, data[:, i])
            else:
                zs[:, idx] = data[:, i]
        return zs

    @classmethod
    def prep(cls):
        feature_file = str(GERMAN_DIR / 'german.features.json')
        feature_spec = json.load(open(feature_file, 'r'))
        filename = str(GERMAN_DIR / 'german.data')

        xs = cls.extract_data(filename, feature_spec)
        np.save(str(GERMAN_DIR / 'german_x.npy'), xs)

        train_xs, test_xs = cls.partition(xs.copy(), ratio=0.7)
        np.save(str(GERMAN_DIR / 'german_x.train.npy'), train_xs)
        np.save('german_x.test.npy', test_xs)

        feature_stats = cls.get_feature_stats(train_xs.copy(), feature_spec)  # Normalize wrt train data to prevent data leak, per X suggestion
        # ! THIS CAUSES THE FEATURES TO BE UNSTABLE!
        json.dump(feature_stats, open(str(GERMAN_DIR / 'german.features.json'), 'w'))

        train_zs = cls.one_hot(cls.normalize(train_xs.copy(), feature_stats), feature_stats)
        test_zs = cls.one_hot(cls.normalize(test_xs.copy(), feature_stats), feature_stats)
        np.save(str(GERMAN_DIR / 'german_z.train.npy'), train_zs)
        np.save(str(GERMAN_DIR / 'german_z.test.npy'), test_zs)

        return xs, train_xs, test_xs, train_zs, test_zs

    @classmethod
    def get_relevant_data(cls):
        # ! THIS FUNCTION IS LIKE PREP, BUT ONLY LOADS!
        feature_file = str(GERMAN_DIR / 'german.features.json')
        feature_spec = json.load(open(feature_file, 'r'))
        filename = str(GERMAN_DIR / 'german.data')

        xs = cls.extract_data(filename, feature_spec)

        train_xs, test_xs = cls.partition(xs.copy(), ratio=0.7)

        feature_stats = cls.get_feature_stats(train_xs.copy(), feature_spec)  # Normalize wrt train data to prevent data leak, per X suggestion

        train_zs = cls.one_hot(cls.normalize(train_xs.copy(), feature_stats), feature_stats)
        test_zs = cls.one_hot(cls.normalize(test_xs.copy(), feature_stats), feature_stats)

        return np.row_stack([train_zs, test_zs]), xs, train_xs, test_xs, train_zs, test_zs


if __name__ == '__main__':
    feature_file = str(GERMAN_DIR / 'german.features.json')
    data_file = str(GERMAN_DIR / 'german.data')

    feature_json = json.load(open(feature_file, 'r'))
    GermanDataset.prep(data_file, feature_json)
