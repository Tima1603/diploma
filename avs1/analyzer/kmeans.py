# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
# import pandas as pd
#
# style.use('ggplot')
#
#
# class K_Means:
#     def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
#         self.k = k
#         self.tolerance = tolerance
#         self.max_iterations = max_iterations
#
#     def fit(self, data):
#
#         self.centroids = {}
#
#         # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
#         for i in range(self.k):
#             self.centroids[i] = data[i]
#
#         # begin iterations
#         for i in range(self.max_iterations):
#             self.classes = {}
#             for i in range(self.k):
#                 self.classes[i] = []
#
#             # find the distance between the point and cluster; choose the nearest centroid
#             for features in data:
#                 distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
#                 classification = distances.index(min(distances))
#                 self.classes[classification].append(features)
#
#             previous = dict(self.centroids)
#
#             # average the cluster datapoints to re-calculate the centroids
#             for classification in self.classes:
#                 self.centroids[classification] = np.average(self.classes[classification], axis=0)
#
#             isOptimal = True
#
#             for centroid in self.centroids:
#
#                 original_centroid = previous[centroid]
#                 curr = self.centroids[centroid]
#
#                 if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
#                     isOptimal = False
#
#             # break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
#             if isOptimal:
#                 break
#
#     def pred(self, data):
#         distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
#         classification = distances.index(min(distances))
#         return classification
#
#
# def main():
#     df = pd.read_csv(r".\data\test.csv")
#     df = df[['size_kb', 'claster_id']]
#     dataset = df.astype(float).values.tolist()
#
#     X = df.values  # returns a numpy array
#
#     km = K_Means(3)
#     km.fit(X)
#
#     # Plotting starts here
#     colors = 10 * ["r", "g", "c", "b", "k"]
#
#     for centroid in km.centroids:
#         plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=130, marker="x")
#
#     for classification in km.classes:
#         color = colors[classification]
#         for features in km.classes[classification]:
#             plt.scatter(features[0], features[1], color=color, s=30)
#
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
# import pandas as pd
#
# style.use('ggplot')
#
#
# class K_Means:
#     def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
#         self.k = k
#         self.tolerance = tolerance
#         self.max_iterations = max_iterations
#
#     def fit(self, data):
#         self.centroids = {}
#         # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
#         for i in range(self.k):
#             self.centroids[i] = data[i]
#
#         # begin iterations
#         for i in range(self.max_iterations):
#             self.classes = {}
#             for i in range(self.k):
#                 self.classes[i] = []
#
#             # find the distance between the point and cluster; choose the nearest centroid
#             for features in data:
#                 distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
#                 classification = distances.index(min(distances))
#                 self.classes[classification].append(features)
#
#             previous = dict(self.centroids)
#
#             # average the cluster datapoints to re-calculate the centroids
#             for classification in self.classes:
#                 self.centroids[classification] = np.average(self.classes[classification], axis=0)
#
#             isOptimal = True
#
#             for centroid in self.centroids:
#                 original_centroid = previous[centroid]
#                 curr = self.centroids[centroid]
#                 if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
#                     isOptimal = False
#
#             # break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
#             if isOptimal:
#                 break
#
#     def pred(self, data):
#         distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
#         classification = distances.index(min(distances))
#         return classification
#
#
# def main():
#     # Load data from test.csv
#     df = pd.read_csv(r".\data\test.csv")
#     # Assuming 'size_kb' and 'claster_id' are relevant columns
#     df = df[['size_kb', 'user_dp_id']]
#     dataset = df.values.tolist()
#
#     X = df.values  # returns a numpy array
#
#     # Initialize K-Means with 3 clusters
#     km = K_Means(3)
#     km.fit(X)
#
#     # Plotting starts here
#     colors = 10 * ["r", "g", "c", "b", "k"]
#
#     for centroid in km.centroids:
#         plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=130, marker="x")
#
#     for classification in km.classes:
#         color = colors[classification]
#         for features in km.classes[classification]:
#             plt.scatter(features[0], features[1], color=color, s=30)
#
#     plt.show()
#
#     # Save clustered data to a CSV file
#     clustered_data = []
#     for classification in km.classes:
#         for features in km.classes[classification]:
#             clustered_data.append(list(features) + [classification])
#
#     clustered_df = pd.DataFrame(clustered_data, columns=['mac_address','size_kb', 'user_dp_id', 'cluster'])
#     clustered_df.to_csv(r".\data\clustered_data.csv", index=False)
#
#
# if __name__ == "__main__":
#     main()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
# import pandas as pd
#
# style.use('ggplot')
#
#
# class K_Means:
#     def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
#         self.k = k
#         self.tolerance = tolerance
#         self.max_iterations = max_iterations
#
#     def fit(self, data):
#         self.centroids = {}
#         # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
#         for i in range(self.k):
#             self.centroids[i] = data[i]
#
#         # begin iterations
#         for i in range(self.max_iterations):
#             self.classes = {}
#             for i in range(self.k):
#                 self.classes[i] = []
#
#             # find the distance between the point and cluster; choose the nearest centroid
#             for features in data:
#                 distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
#                 classification = distances.index(min(distances))
#                 self.classes[classification].append(features)
#
#             previous = dict(self.centroids)
#
#             # average the cluster datapoints to re-calculate the centroids
#             for classification in self.classes:
#                 self.centroids[classification] = np.average(self.classes[classification], axis=0)
#
#             isOptimal = True
#
#             for centroid in self.centroids:
#                 original_centroid = previous[centroid]
#                 curr = self.centroids[centroid]
#                 if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
#                     isOptimal = False
#
#             # break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
#             if isOptimal:
#                 break
#
#     def pred(self, data):
#         distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
#         classification = distances.index(min(distances))
#         return classification
#
#
# def main():
#     # Load data from test.csv
#     df = pd.read_csv(r".\data\boxcoxed_data1.csv")
#     # Assuming 'size_kb' and 'claster_id' are relevant columns
#     df = df[['size_kb', '@timestamp']]
#     dataset = df.values.tolist()
#
#     X = df.values  # returns a numpy array
#
#     # Initialize K-Means with 3 clusters
#     km = K_Means(3)
#     km.fit(X)
#
#     # Plotting starts here
#     colors = 10 * ["r", "g", "c", "b", "k"]
#
#     for centroid in km.centroids:
#         plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=130, marker="x")
#
#     for classification in km.classes:
#         color = colors[classification]
#         for features in km.classes[classification]:
#             plt.scatter(features[0], features[1], color=color, s=30)
#
#     plt.title('3 Кластера')
#     plt.xlabel('size_kb')
#     plt.ylabel('@timestamp')
#     plt.show()
#
#     # Save clustered data to a CSV file
#     clustered_data = []
#     for classification in km.classes:
#         for features in km.classes[classification]:
#             clustered_data.append(list(features) + [classification])
#
#     clustered_df = pd.DataFrame(clustered_data, columns=['size_kb', '@timestamp', 'cluster'])
#     clustered_df.to_csv(r".\data\cluster.csv", index=False)
#
#
# if __name__ == "__main__":
#     main()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
# import pandas as pd
#
# style.use('ggplot')
# class K_Means:
#     def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
#         self.k = k
#         self.tolerance = tolerance
#         self.max_iterations = max_iterations
#
#     def fit(self, data):
#         self.centroids = {}
#         # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
#         for i in range(self.k):
#             self.centroids[i] = data[i]
#
#         # begin iterations
#         for i in range(self.max_iterations):
#             self.classes = {}
#             for i in range(self.k):
#                 self.classes[i] = []
#
#             # find the distance between the point and cluster; choose the nearest centroid
#             for features in data:
#                 distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
#                 classification = distances.index(min(distances))
#                 self.classes[classification].append(features)
#
#             previous = dict(self.centroids)
#
#             # average the cluster datapoints to re-calculate the centroids
#             for classification in self.classes:
#                 self.centroids[classification] = np.average(self.classes[classification], axis=0)
#
#             isOptimal = True
#
#             for centroid in self.centroids:
#                 original_centroid = previous[centroid]
#                 curr = self.centroids[centroid]
#                 if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
#                     isOptimal = False
#
#             # break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
#             if isOptimal:
#                 break
# #
#     def pred(self, data):
#         distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
#         classification = distances.index(min(distances))
#         return classification
#
#
# def main():
#     # Load data from test.csv
#     df = pd.read_csv(r".\data\boxcoxed_data.csv")
#     # Assuming 'size_kb' and 'claster_id' are relevant columns
#     df = df[['size_kb', '@timestamp']]
#     dataset = df.values.tolist()
#
#     X = df.values  # returns a numpy array
#
#     # Initialize K-Means with 3 clusters
#     km = K_Means(3)
#     km.fit(X)
#
#     # Plotting starts here
#     colors = 10 * ["r", "g", "c", "b", "k"]
#
#     for centroid in km.centroids:
#         plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=130, marker="x")
#
#     for classification in km.classes:
#         color = colors[classification]
#         for features in km.classes[classification]:
#             plt.scatter(features[0], features[1], color=color, s=30)
#
#     plt.title('3 Кластера')
#     plt.xlabel('size_kb')
#     plt.ylabel('@timestamp')
#     plt.show()
#
#     # Save clustered data to a CSV file
#     clustered_data = []
#     for classification in km.classes:
#         for features in km.classes[classification]:
#             clustered_data.append(list(features) + [classification])
#
#     clustered_df = pd.DataFrame(clustered_data, columns=['size_kb', '@timestamp', 'cluster'])
#     clustered_df.to_csv(r".\data\clustered_data.csv", index=False)
#
#
# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from scipy import stats
from sklearn.preprocessing import StandardScaler

style.use('ggplot')


class K_Means:
    def __init__(self, input_file, output_file, k=3, tolerance=0.0001, max_iterations=500):
        self.input_file = input_file
        self.output_file = output_file
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def preprocess_data(self):
        # Load data
        df = pd.read_csv(self.input_file)
        df['size_kb'], _ = stats.boxcox(df['size_kb'])
        df['@timestamp'], _ = stats.boxcox(pd.to_datetime(df['@timestamp']).astype('int64'))

        # Normalize data using StandardScaler
        scaler = StandardScaler()
        columns_to_normalize = ['size_kb', '@timestamp']
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

        # Save preprocessed data
        df.to_csv(self.output_file, index=False)

        return df

    def fit(self):
        df = self.preprocess_data()

        # Assuming 'size_kb' and '@timestamp' are relevant columns
        X = df[['size_kb', '@timestamp']].values

        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = X[i]

        for _ in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []

            for features in X:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)

            previous = dict(self.centroids)

            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis=0)

            is_optimal = True

            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    is_optimal = False

            if is_optimal:
                break

    def plot_clusters(self):
        colors = 10 * ["r", "g", "c", "b", "k"]

        for centroid in self.centroids:
            plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1], s=130, marker="x")

        for classification in self.classes:
            color = colors[classification]
            for features in self.classes[classification]:
                plt.scatter(features[0], features[1], color=color, s=30)

        plt.title('3 Clusters')
        plt.xlabel('size_kb')
        plt.ylabel('@timestamp')
        plt.show()

    def save_clusters(self):
        clustered_data = []
        for classification in self.classes:
            for features in self.classes[classification]:
                clustered_data.append(list(features) + [classification])

        clustered_df = pd.DataFrame(clustered_data, columns=['size_kb', '@timestamp', 'cluster'])
        clustered_df.to_csv(self.output_file, index=False)


# Пример использования класса в другом файле
# from kmeans_clusterer import KMeansClusterer

# def main():
#     input_file = r".\data\data.csv"
#     output_file = r".\data\boxcoxed_data.csv"
#
#     kmeans = KMeansClusterer(input_file, output_file)
#     kmeans.fit()
#     kmeans.plot_clusters()
#     kmeans.save_clusters()
#
#
# if __name__ == "__main__":
#     main()



