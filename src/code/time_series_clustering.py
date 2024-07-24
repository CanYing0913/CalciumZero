import math
import random
import time
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans

from src.utils import iprint


def plot_series(data, save_dir):
    fig, axs = plt.subplots(259, 4, figsize=(10, 250))
    for i in range(259):
        for j in range(4):
            index = i * 4 + j
            if index >= len(data):
                continue
            # Plot the series in the current subplot
            axs[i, j].plot(data[index])
    plt.savefig(path.join(save_dir, 'plot_series.jpg'), format='jpg', dpi=300)
    plt.show()
    plt.close()  # Close the figure to free up memory


def nan_counter(list_of_series):
    nan_polluted_series_counter = 0
    for series in list_of_series:
        if pd.Series(series).isnull().sum() > 0:
            nan_polluted_series_counter += 1
    print(nan_polluted_series_counter)


def scale_data(data):
    scaler = None
    for i in range(len(data)):
        scaler = MinMaxScaler()
        data[i] = scaler.fit_transform(np.array(data[i]).reshape(-1, 1)).flatten()
    print("max: "+str(max(data[0]))+"\tmin: "+str(min(data[0])))
    print(data[0][:5])
    return data, scaler


class SOMClass:
    def __init__(self, scaled_data, data, namesofMySeries, scaler):
        self.data = scaled_data
        self.original_data = data
        self.namesofMySeries = namesofMySeries
        self.scaler = scaler
        self.som_x = self.som_y = math.ceil(math.sqrt(math.sqrt(len(self.data)))/2)
        self.som = MiniSom(self.som_x, self.som_y, len(self.data[0]), sigma=0.3, learning_rate=0.1)
        self.df = None  # Initialize DataFrame as None
        self.averages_df = None
        self.scale_averages_df = None

    def win_map(self):
        return self.som.win_map(self.data)

    def train(self):
        start = time.time()
        self.som.random_weights_init(self.data)
        self.som.train(self.data, 50000)
        end = time.time()
        print(f"SOM Training Time: {end - start} seconds")

    def plot_som_series_center(self, save_dir, max_series=5):
        win_map = self.win_map()

        fig, axs = plt.subplots(self.som_x, self.som_y, figsize=(50, 25))
        fig.suptitle('Clusters for SOM', fontsize=16)
        scale_average = []
        for x in range(self.som_x):
            for y in range(self.som_y):
                cluster = (x, y)
                if cluster in win_map.keys():
                    # plot selected series
                    num_series = min(max_series, len(win_map[cluster]))

                    # Randomly select series to plot
                    print(cluster)
                    selected_series = random.sample(win_map[cluster], num_series)
                    for i, series in enumerate(selected_series):
                        axs[cluster].plot(series, c="gray", alpha=0.5)
                        print(i)

                    # Arithmetic mean
                    # axs[cluster].plot(np.average(np.vstack(self.win_map[cluster]), axis=0), c="red")
                    # Dynamic Time Warping Barycenter Averaging
                    average_temp = dtw_barycenter_averaging(np.vstack(win_map[cluster]))
                    axs[cluster].plot(average_temp, c="red")
                    scale_average.append(average_temp.flatten())

                cluster_number = x * self.som_y + y + 1
                axs[cluster].set_title(f"Cluster {cluster_number}", fontsize=16)

        # Save the DataFrame to a CSV file
        self.scale_averages_df = pd.DataFrame(scale_average).transpose()
        # output_path = "../data/scale_cluster_average.txt"  # Define the path and filename
        # scale_cluster_average.to_csv(output_path)
        plt.savefig(path.join(save_dir, 'som_clusters.jpg'), format='jpg', dpi=300)
        plt.show()
        plt.close(fig)  # Close the figure to free up memory

    def cluster_distribution(self, save_dir):
        win_map = self.win_map()
        cluster_c = []
        cluster_n = []
        for x in range(self.som_x):
            for y in range(self.som_y):
                cluster = (x, y)
                if cluster in win_map.keys():
                    cluster_c.append(len(win_map[cluster]))
                else:
                    cluster_c.append(0)
                cluster_number = x * self.som_y + y + 1
                cluster_n.append(f"Cluster {cluster_number}")
        plt.figure(figsize=(25, 5))
        plt.title("Cluster Distribution for SOM")
        plt.bar(cluster_n, cluster_c)
        plt.savefig(path.join(save_dir, 'cluster_distribution.jpg'), format='jpg', dpi=300)
        plt.show()
        plt.close()  # Close the figure to free up memory

    def cluster_mapping(self, save_dir):
        cluster_map = []
        for idx in range(len(self.data)):
            winner_node = self.som.winner(self.data[idx])
            cluster_map.append(
                (self.namesofMySeries[idx], f"Cluster {winner_node[0] * self.som_y + winner_node[1] + 1}"))
        #     cluster_map.append((idx, f"Cluster {winner_node[0]*som_y + winner_node[1] + 1}"))
        self.df = pd.DataFrame(cluster_map, columns=["Series", "Cluster"]).sort_values(by="Cluster").set_index("Series")
        # Save the DataFrame to a CSV file
        output_path = path.join(save_dir, "cluster_map.csv")  # Define the path and filename
        self.df.to_csv(output_path)

    def dtw_average(self):
        # Group series by cluster
        cluster_groups = self.df.groupby('Cluster').apply(lambda x: list(x.index))
        # Extract numeric part and convert to integer
        cluster_groups.index = pd.to_numeric(cluster_groups.index.str.extract('(\d+)')[0])
        cluster_groups.sort_index(inplace=True)

        # Calculate DTW Barycenter for each cluster
        cluster_averages = {}
        for cluster_id, indices in cluster_groups.items():
            print(cluster_id, indices)
            series_in_cluster = [self.original_data[idx-1] for idx in indices]
            if len(series_in_cluster) > 1:  # DBA needs at least two series to compute
                average_series = dtw_barycenter_averaging(np.vstack(series_in_cluster))
                print(average_series)
                cluster_averages[cluster_id] = average_series.flatten()
            else:
                # If only one series in the cluster, that series is the "average"
                cluster_averages[cluster_id] = series_in_cluster[0].flatten()

        # Optionally, convert cluster averages to a DataFrame or similar structure for further analysis
        self.averages_df = pd.DataFrame.from_dict(cluster_averages)

    # def save_averages_df(self):
    #     # Generate the header string with "Time" and column names
    #     num_columns = len(self.averages_df.columns)
    #     header = "TIME\t" + "\t".join([f"ROI_{i}" for i in range(1, num_columns)]) + "\n"
    #     # Generate the index numbers for the "Time" column
    #     index_numbers = "\n".join(
    #         [str(i) + "\t" + "\t".join(map(str, row)) for i, row in enumerate(self.averages_df.values, 1)]) + "\n"
    #     # Write the header followed by the index numbers to a text file
    #     with open(f"../data/cluster_average.txt", "w") as file:
    #         file.write(header + index_numbers)

    @staticmethod
    def save_df_to_txt(df, filename):
        # Generate the header string with "Time" and column names
        num_columns = len(df.columns)
        header = "TIME\t" + "\t".join([f"ROI_{i}" for i in range(1, num_columns + 1)]) + "\n"

        # Generate the index numbers for the "Time" column
        index_numbers = "\n".join(
            [str(i) + "\t" + "\t".join(map(str, row)) for i, row in enumerate(df.values, 1)]) + "\n"

        # Write the header followed by the index numbers to a text file
        with open(filename, "w") as file:
            file.write(header + index_numbers)

    def save_averages_df(self):
        self.save_df_to_txt(self.averages_df, "../data/cluster_average.txt")

    def save_scale_averages_df(self):
        self.save_df_to_txt(self.scale_averages_df, "../data/scale_cluster_average.txt")

    def run(self):
        self.train()
        self.plot_som_series_center()
        self.cluster_distribution()
        self.cluster_mapping()
        self.dtw_average()
        self.save_averages_df()
        self.save_scale_averages_df()


class KMeansClass:
    def __init__(self, data, namesofMySeries):
        self.data = data
        self.namesofMySeries = namesofMySeries
        self.kmeans_x = self.kmeans_y = math.ceil(math.sqrt(math.sqrt(len(self.data))))
        # self.cluster_count = math.ceil(math.sqrt(len(self.data)))
        self.cluster_count = self.kmeans_x * self.kmeans_y
        self.model = TimeSeriesKMeans(n_clusters=self.cluster_count, metric="dtw")

    def fit_predict(self):
        start = time.time()
        self.labels = self.model.fit_predict(self.data)
        end = time.time()
        print(f"KMeans Clustering Time: {end - start} seconds")
        return self.labels

    def plot_kmeans_series_center(self):
        plot_count = math.ceil(math.sqrt(self.cluster_count))
        fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
        fig.suptitle('Clusters for KMeans')
        row_i = 0
        column_j = 0
        for label in set(self.labels):
            cluster = []
            for i in range(len(self.labels)):
                if (self.labels[i] == label):
                    axs[row_i, column_j].plot(self.data[i], c="gray", alpha=0.4)
                    cluster.append(self.data[i])
            if len(cluster) > 0:
                # Arithmetic mean
                # axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
                # Dynamic Time Warping Barycenter Averaging
                axs[row_i, column_j].plot(dtw_barycenter_averaging(np.vstack(cluster)), c="red")
            axs[row_i, column_j].set_title("Cluster " + str(row_i * self.kmeans_y + column_j + 1))
            column_j += 1
            if column_j % plot_count == 0:
                row_i += 1
                column_j = 0
        plt.show()

    def plot_distribution(self):
        cluster_c = [len(self.labels[self.labels == i]) for i in range(self.cluster_count)]
        cluster_n = ["Cluster " + str(i) for i in range(self.cluster_count)]
        plt.figure(figsize=(15, 5))
        plt.title("Cluster Distribution for KMeans")
        plt.bar(cluster_n, cluster_c)
        plt.show()

    def cluster_mapping(self):
        fancy_names_for_labels = [f"Cluster {label}" for label in self.labels]
        df = pd.DataFrame(zip(self.namesofMySeries, fancy_names_for_labels), columns=["Series", "Cluster"]).sort_values(
            by="Cluster").set_index("Series")
        print(df)


def ts_clustering(filepath='src/code/G2_data.npy'):
    # Load data
    input_data = np.load(filepath)
    input_data_list = input_data.tolist()
    namesofMySeries = range(1, len(input_data_list) + 1)

    # show series
    # plot_series(input_data_list)

    # check series length and nan values
    series_lengths = {len(series) for series in input_data_list}
    # print(series_lengths)
    # print(nan_counter(input_data_list))

    # select test data range
    # data = input_data_list[:50] # test
    data = input_data_list.copy()
    # process to scale data
    scaled_data, scaler = scale_data(data)
    # print(scaler)
    # Self-organizing map(SOM) clustering
    som = SOMClass(scaled_data, input_data, namesofMySeries, scaler)
    som.run()

    # K-means clustering
    # kmeans = KMeansClass(data, namesofMySeries)
    # kmeans.fit_predict()
    # kmeans.plot_kmeans_series_center()
    # kmeans.plot_distribution()
    # kmeans.cluster_mapping()

