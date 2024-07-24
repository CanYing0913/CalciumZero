import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import seaborn as sns
import pandas as pd


def load_data(filepath):
    """Load data from a file and return all columns as a DataFrame."""
    return pd.read_csv(filepath, header=0, delimiter="\t")


def show_signal(data, window_size):
    # detection
    algo = rpt.Pelt(model="rbf").fit(data)
    result = algo.predict(pen=window_size)
    # display
    rpt.display(data, result)
    plt.show()


def calculate_moving_average(data, window_size=50):
    """Calculate and return the moving average of the provided data."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def perform_change_point_detection(data, penalty):
    """Detect and return change points in the provided data."""
    algo = rpt.Pelt(model="rbf").fit(data)
    return algo.predict(pen=penalty)


def plot_data_with_changes(data, moving_avg, change_points, title='Change Detection Plot', save_path='plot.png'):
    """Plot data, its moving average, and change points with filled color segments and save the plot."""
    time_axis = np.arange(len(data))
    colors = sns.color_palette('Set1', n_colors=len(change_points) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, data, label='Data', alpha=0.5)
    plt.plot(time_axis, moving_avg, label='Moving Average', color='green')

    for i, (start, end) in enumerate(zip([0] + change_points, change_points + [len(data)])):
        segment = data[start:end]
        if segment.size > 0:
            lower, upper = np.min(segment), np.max(segment)
            plt.fill_between(time_axis[start:end], lower, upper, color=colors[i], alpha=0.5, label=f'Phase {i + 1}')

    plt.xlabel('Time')
    plt.ylabel('Height')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)  # Save the plot to a file
    plt.close()  # Close the figure to free memory


def save_list_to_txt(data_list, filename):
    with open(filename, 'w') as file:
        for item in data_list:
            file.write(str(item) + '\n')


def tl_detect():
    # Parameters
    cluster = "cluster_average"
    scale_cluster = "scale_cluster_average"
    filepath = scale_cluster
    window_size = 50
    penalty = 25

    # Load data and perform operations for each column
    data = load_data(f'../data/{filepath}.txt')
    change_point_list = []
    for column_name in data.columns[1:]:  # Skip the first column (assuming it's 'TIME')
        column_data = data[column_name].to_numpy()
        # signal = show_signal(column_data, window_size)
        moving_avg = calculate_moving_average(column_data, window_size)
        change_points = perform_change_point_detection(column_data, penalty)
        print(change_points)
        change_point_list.append(change_points)
        # plot_data_with_changes(column_data, moving_avg, change_points, title=f'Change Detection Plot for {
        # column_name}', save_path='./plot')
        title = f'Change Detection Plot for Cluster {column_name[4:]}'
        save_path = f'../plot/time_lapse_plot/{filepath[:-8]}/plot_{column_name}.png'
        plot_data_with_changes(column_data, moving_avg, change_points, title=title, save_path=save_path)

    save_list_to_txt(change_point_list, f'../data/{filepath[:-8]}_phases.txt')

