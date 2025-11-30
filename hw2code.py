import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

def find_best_split(feature_vector, target_vector):
    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)

    order = np.argsort(feature_vector)
    feature_sorted = feature_vector[order]
    target_sorted = target_vector[order]

    diffs = np.diff(feature_sorted)
    valid = diffs != 0

    if not np.any(valid):
        return np.array([]), np.array([]), None, None

    thresholds = (feature_sorted[:-1] + feature_sorted[1:]) / 2
    thresholds = thresholds[valid]

    target_bin = target_sorted.astype(int)

    cumsum = np.cumsum(target_bin)

    total_pos = cumsum[-1]
    total = len(target_bin)

    split_indices = np.where(valid)[0]

    n_left = split_indices + 1
    n_right = total - n_left

    pos_left = cumsum[split_indices]
    pos_right = total_pos - pos_left

    p1_left = pos_left / n_left
    p0_left = 1 - p1_left
    H_left = 1 - p1_left**2 - p0_left**2

    p1_right = pos_right / n_right
    p0_right = 1 - p1_right
    H_right = 1 - p1_right**2 - p0_right**2

    ginis = -(n_left / total) * H_left - (n_right / total) * H_right

    best_idx = np.argmax(ginis)

    return thresholds, ginis, thresholds[best_idx], ginis[best_idx]



class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
        return {
            'feature_types': self.feature_types,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf
        }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self.max_depth is not None and depth >= self.max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self.min_samples_split is not None and len(sub_y) < self.min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    if current_count > 0:
                        ratio[key] = current_click / current_count
                    else:
                        ratio[key] = 0.0
                sorted_categories = sorted(ratio.items(), key=lambda x: x[1])
                sorted_categories = [x[0] for x in sorted_categories]
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini is not None and (gini_best is None or gini > gini_best):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [x[0] for x in categories_map.items() if x[1] < threshold]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self.min_samples_leaf is not None:
            if np.sum(split) < self.min_samples_leaf or np.sum(~split) < self.min_samples_leaf:
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature_idx = node["feature_split"]
        feature_type = self.feature_types[feature_idx]
        
        if feature_type == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._tree = {}  # Сбрасываем дерево перед обучением
        self._fit_node(X, y, self._tree, depth=0)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)