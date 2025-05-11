from collections import Counter


class KNN:
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

    def __init__(self, k=3, distance=EUCLIDEAN, x_train=None, y_train=None):
        if distance not in (self.EUCLIDEAN, self.MANHATTAN):
            raise ValueError(f"Unsupported distance metric: {distance}")
        self.k = k
        self.distance = distance
        self.x_train = x_train
        self.y_train = y_train

        # Mapping distance names to methods
        self.distances = {
            self.EUCLIDEAN: self.__euclidean_distance,
            self.MANHATTAN: self.__manhattan_distance
        }

    def __euclidean_distance(self, test_data, train_data):
        distance = 0
        for i, val in enumerate(test_data):
            distance += (val - train_data[i]) ** 2
        return distance ** 0.5

    def __manhattan_distance(self, test_data, train_data):
        distance = 0
        for i, val in enumerate(test_data):
            distance += abs(val - train_data[i])
        return distance

    def get_neighbors(self, test_point):

        distance_func = self.distances[self.distance]
        distances = [
            (distance_func(test_point, train_point), label)
            for train_point, label in zip(self.x_train, self.y_train)
        ]
        return distances

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for test_point in x_test:
            distances = self.get_neighbors(test_point)

            distances.sort(key=lambda x: x[0])

            k_labels = [label for _, label in distances[:self.k]]
            label_counts = Counter(k_labels)
            most_common = label_counts.most_common()
            top_count = most_common[0][1]
            tied_labels = [label for label,
                           count in most_common if count == top_count]

            if len(tied_labels) == 1:
                predictions.append(tied_labels[0])
            else:
                # Tie: use inverse squared distance weighting
                weights = {label: 0.0 for label in tied_labels}
                for dist, label in distances[:self.k]:
                    if label in weights:
                        # Add small epsilon to avoid div by zero
                        weights[label] += 1 / ((dist + 1e-7) ** 2)
                best_label = max(weights.items(), key=lambda x: x[1])[0]
                predictions.append(best_label)
        return predictions
