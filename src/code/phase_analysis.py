import pandas as pd


# Define function to generate dataframe for each ROI based on phases
def generate_phase_dataframe(roi_data, phase_end_times):
    phase_dfs = []
    start_time = 0
    i = 0  # phase counter

    for end_time in phase_end_times:
        phase_data = roi_data[(roi_data['Time'] >= start_time) & (roi_data['Time'] < end_time)]
        frequency = phase_data.shape[0] / (end_time - start_time)
        baseline = phase_data['Height'].min()
        rise_time = phase_data['Rise_Time'].sum()
        decay_time = phase_data['Fall_Time'].sum()
        phase_df = pd.DataFrame({
            'Phase': f'Phase {i}',
            'Frequency': [frequency],
            'Baseline (calcium)': [baseline],
            'Rise Time': [rise_time],
            'Decay Time': [decay_time]
        })
        phase_dfs.append(phase_df)

        start_time = end_time
        i += 1

    result_df = pd.concat(phase_dfs).reset_index(drop=True)
    return result_df


# # Filter the dataset for ROI 1
# roi_1_data = average_peak_data[average_peak_data['ROI(#)'] == 1]
# roi_1_phases = phase_times[0]  # Get end times for phases of ROI 1
#
# # Generate the DataFrame for ROI 1
# roi_1_dataframe = generate_phase_dataframe(roi_1_data, roi_1_phases)
# print(roi_1_dataframe)

# Iterate over all ROIs based on their number in the phase_times list


def analysis():
    # Load the CSV data
    average_peak_data_path = "../peakcaller/scale_cluster_average_Peak_Data.csv"
    average_peak_data = pd.read_csv(average_peak_data_path)

    # Read phase segmentation data from the text file
    phases_path = "../data/scale_cluster_phases.txt"
    with open(phases_path, 'r') as file:
        phase_lines = file.readlines()

    # Convert each line to a list of integers (segment start times)
    phase_times = [list(map(int, line.strip().replace('[', '').replace(']', '').split(','))) for line in phase_lines]
    print(phase_times)
    for index, roi_phases in enumerate(phase_times):
        roi_number = index + 1  # ROI numbers are 1-based
        roi_data = average_peak_data[average_peak_data['ROI(#)'] == roi_number]
        roi_dataframe = generate_phase_dataframe(roi_data, roi_phases)
        roi_dataframe[['Frequency', 'Baseline (calcium)']] = roi_dataframe[
            ['Frequency', 'Baseline (calcium)']].applymap(
            lambda x: f"{x:e}")
        print(roi_dataframe)
        # Save each dataframe to a CSV file
        file_path = f"../data/phase/ROI_{roi_number}_data.csv"
        roi_dataframe.to_csv(file_path)
        print(f"Data for ROI {roi_number} saved to {file_path}")
