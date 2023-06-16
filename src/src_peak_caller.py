"""
Source file for Section 4 - Peak Caller
Last edited on Jun 14, 2023
Author: Xuchen Wang (xw2747@columbia.edu), Yian Wang (canying0913@gmail.com)
For all inquiry, please contact Xuchen Wang.
Copyright Yian Wang - 2023
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import linalg as LA
from numpy import mean, absolute
from numpy.linalg import matrix_power
from scipy.signal import find_peaks


# The only functions a user would call(Current Version) are
# Detrender_2(does nothing at all)
# Find_Peak(finds the peaks)
# Print_ALL_Peaks(generate figure for all peaks)
# Raster_Plot(generate a dot plot, not raster plot since it would be messy and no different from a dot plot)
# Histogram_Height(generate a histogram of heights)
# Histogram_Time(generate a histogram of time)
# Save_Result
# Synchronization
# Correlation


def mad(data, axis=None):
    return mean(absolute(data - mean(data, axis)), axis)


def flatten(lst):
    return [item for sublist in lst for item in sublist]


class PeakCaller:
    def __init__(self, seq, filename):
        self.num_peak_rec = 0
        self.seq = seq
        self.filename = Path(filename).with_suffix('')
        self.obs_num = len(seq)
        self.length = len(seq[0])
        self.smoothed_seq = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.detrended_seq = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.peak_start = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.peak_half_start = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.peak_end = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.peak_half_end = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.peak_loc = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.filterer_peak_loc = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.filterer_peak_loc_2 = [[] for _ in range(self.obs_num)]
        self.filterer_peak_height_mean = [0 for _ in range(self.obs_num)]
        self.filterer_peak_height = [[] for _ in range(self.obs_num)]
        self.filterer_peak_rise_time = [[] for _ in range(self.obs_num)]
        self.filterer_peak_fall_time = [[] for _ in range(self.obs_num)]
        self.filterer_peak_half_start = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.filterer_peak_half_end = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.std_after_removal = [0 for _ in range(self.obs_num)]
        self.peak_height = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.peak_height_std = [0 for _ in range(self.obs_num)]
        self.peak_height_mean = [0 for _ in range(self.obs_num)]
        self.peak_rise_time = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.peak_fall_time = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        self.non_peak_std = [0 for _ in range(self.obs_num)]
        self.series_std = [0 for _ in range(self.obs_num)]
        self.series_mad = [0 for _ in range(self.obs_num)]
        self.series_rel_std = [0 for _ in range(self.obs_num)]
        self.series_rel_std_sorted = [[0, i] for i in range(self.obs_num)]
        self.matrix_smoother = np.ones((self.length, self.length)) / self.length
        self.candidate_mean_prominence = [0 for _ in range(self.obs_num)]
        self.peak_mean_prominence = [0 for _ in range(self.obs_num)]
        self.peak_std_prominence = [0 for _ in range(self.obs_num)]
        self.TrendSmoothness = 25

    def Detrender(self, mark=0, s=60):
        if mark == 1:
            base_mat = np.diag([-2 for i in range(self.length)]) + np.diag([1 for i in range(self.length - 1)],
                                                                           1) + np.diag(
                [1 for i in range(self.length - 1)], -1)
            base_mat[0, 1] = 2
            base_mat[self.length - 1, self.length - 2] = 2
            base_mat = base_mat / 4 + np.identity(self.length)
            self.matrix_smoother = matrix_power(base_mat, 4 * s)
        for i in range(self.obs_num):
            self.smoothed_seq[i] = np.matmul(self.matrix_smoother, self.seq[i])
        self.detrended_seq = np.divide(self.seq, np.abs(self.smoothed_seq))
        for j in range(self.obs_num):
            self.series_std[j] = np.std(self.detrended_seq[j])
            self.series_rel_std[j] = self.series_std[j] / (
                        np.max(self.detrended_seq[j]) - np.min(self.detrended_seq[j]))
            self.series_rel_std_sorted[j][0] = self.series_rel_std[j]
        self.series_rel_std_sorted.sort()

    def Detrender_2(self):
        self.detrended_seq = self.seq
        for j in range(self.obs_num):
            self.series_std[j] = np.std(self.detrended_seq[j])
            self.series_mad[j] = mad(self.detrended_seq[j])
            self.series_rel_std[j] = self.series_std[j] / (
                        np.max(self.detrended_seq[j]) - np.min(self.detrended_seq[j]))
            self.series_rel_std_sorted[j][0] = self.series_rel_std[j]
        self.series_rel_std_sorted.sort()

    def Find_Peak_2(self, lookafter=25, lookbefore=25, rise=16.0, fall=16.0):
        rise_ratio = (rise - 1) / 100
        fall_ratio = (fall - 1) / 100
        candidate = [[] for _ in range(self.obs_num)]
        pks = [[] for _ in range(self.obs_num)]
        for i in range(self.obs_num):
            candidate[i], properties = find_peaks(data[i], prominence=(30))
            self.candidate_mean_prominence[i] = np.mean(properties['prominences'])
            peak_prominence_lst = []
            required_rise = rise_ratio
            required_fall = fall_ratio
            prior_peak = 0
            Range = (np.max(self.detrended_seq[i]) - np.min(self.detrended_seq[i]))
            for j in range(len(candidate[i])):
                k = candidate[i][j]
                if k - lookbefore < prior_peak:
                    continue
                # lookbackindex=max(prior_peak,k-lookbefore)
                dropit = 0
                lookbackindex = max(0, k - lookbefore)
                minbefore = min(self.detrended_seq[i][lookbackindex:k + 1])
                min_bf_index = np.argmin(self.detrended_seq[i][lookbackindex:k + 1]) + lookbackindex
                if minbefore < self.detrended_seq[i][k] - Range * required_rise:
                    lookaheadthresh = min(self.length - 1, k + lookafter)
                    lookaheadindex = lookaheadthresh
                    for afterindex in range(k + 1, lookaheadthresh + 1):
                        if self.detrended_seq[i][k] < self.detrended_seq[i][afterindex]:
                            lookaheadindex = afterindex
                            dropit = 1
                            break
                    if dropit == 1:
                        continue
                    minafter = min(self.detrended_seq[i][k:lookaheadindex + 1])
                    min_af_index = np.argmin(self.detrended_seq[i][k:lookaheadindex + 1]) + k
                    if minafter < self.detrended_seq[i][k] - Range * required_rise:
                        peak_prominence_lst.append(properties['prominences'][j])
                        self.peak_loc[i][k] = 1
                        self.peak_start[i][k] = min_bf_index
                        self.peak_half_start[i][k] = np.where(
                            self.detrended_seq[i][min_bf_index:k + 1] <= (minbefore + self.detrended_seq[i][k]) / 2)[0][
                                                         -1] + min_bf_index
                        self.peak_end[i][k] = min_af_index
                        self.peak_half_end[i][k] = np.where(
                            self.detrended_seq[i][k:min_af_index + 1] <= (minafter + self.detrended_seq[i][k]) / 2)[0][
                                                       0] + k
                        self.peak_rise_time[i][k] = k - self.peak_half_start[i][k]
                        self.peak_fall_time[i][k] = self.peak_half_end[i][k] - k
                        height = (2 * self.detrended_seq[i][k] - minbefore - minafter) / 2
                        # height=max(self.detrended_seq[i][k]-minbefore,self.detrended_seq[i][k]-minafter)
                        pks[i].append(height)
                        self.peak_height[i][k] = height
                        prior_peak = k
            next_peak = self.length - 1
            self.peak_height_std[i] = np.std(np.array(pks[i]))
            self.peak_height_mean[i] = np.mean(np.array(pks[i]))
            self.peak_mean_prominence[i] = np.mean(peak_prominence_lst)
            self.peak_std_prominence[i] = np.std(peak_prominence_lst)
            continue
            for k in reversed(candidate[i]):
                lookafterindex = min(next_peak, k + lookafter)
                minafter = min(self.detrended_seq[i][k:lookafterindex + 1])
                if minafter < (1 - required_rise) * self.detrended_seq[i][k]:
                    next_peak = k
                else:
                    self.peak_loc[i][k] = 0
        for num in range(self.obs_num):
            loc = np.where(self.peak_height[num] > 0)[0]
            # loc=np.where((self.peak_height[num]>self.peak_height_mean[num]+3*self.peak_height_std[num]))[0]
            self.filterer_peak_loc[num][loc] = 1
            self.filterer_peak_half_start[num][self.peak_half_start[num][loc].astype(int)] = 1
            self.filterer_peak_half_end[num][self.peak_half_end[num][loc].astype(int)] = 1
            self.filterer_peak_loc_2[num] = loc
            heights = self.peak_height[num][loc]
            rise_times = self.peak_rise_time[num][loc]
            fall_times = self.peak_fall_time[num][loc]
            self.filterer_peak_height_mean[num] = np.mean(heights)
            self.filterer_peak_height[num] = list(heights)
            self.filterer_peak_rise_time[num] = list(rise_times)
            self.filterer_peak_fall_time[num] = list(fall_times)
        for num in range(self.obs_num):
            index_lst = [1 for _ in range(self.obs_num)]
            for ind in self.filterer_peak_loc[num]:
                if ind == 1:
                    for i in range(int(self.peak_start[num][i]), int(self.peak_end[num][i] + 1)):
                        index_lst[i] = 0
            real_index = np.where(np.array(index_lst) == 1)
            other_points = self.detrended_seq[num][real_index]
            self.non_peak_std[num] = np.std(other_points)

    def Find_Peak(self, lookafter=25, lookbefore=25, rise=16.0, fall=16.0):
        rise_ratio = (rise - 1) / 100
        fall_ratio = (fall - 1) / 100
        candidate = [[] for _ in range(self.obs_num)]
        pks = [[] for _ in range(self.obs_num)]
        for i in range(self.obs_num):
            for j in range(self.length):
                if j == 0 and self.detrended_seq[i][j] > self.detrended_seq[i][j + 1]:
                    candidate[i].append(j)
                elif j == self.length - 1 and self.detrended_seq[i][j] > self.detrended_seq[i][j - 1]:
                    candidate[i].append(j)
                elif j != 0 and j != self.length - 1 and self.detrended_seq[i][j] > self.detrended_seq[i][j + 1] and \
                        self.detrended_seq[i][j] > self.detrended_seq[i][j - 1]:
                    candidate[i].append(j)
            required_rise = rise_ratio
            required_fall = fall_ratio
            prior_peak = 0
            for k in candidate[i]:
                if k - lookbefore < prior_peak:
                    continue
                # lookbackindex=max(prior_peak,k-lookbefore)
                dropit = 0
                lookbackindex = max(0, k - lookbefore)
                minbefore = min(self.detrended_seq[i][lookbackindex:k + 1])
                min_bf_index = np.argmin(self.detrended_seq[i][lookbackindex:k + 1]) + lookbackindex
                if minbefore < (1 - required_rise) * self.detrended_seq[i][k]:
                    lookaheadthresh = min(self.length - 1, k + lookafter)
                    lookaheadindex = lookaheadthresh
                    for afterindex in range(k + 1, lookaheadthresh + 1):
                        if self.detrended_seq[i][k] < self.detrended_seq[i][afterindex]:
                            lookaheadindex = afterindex
                            dropit = 1
                            break
                    if dropit == 1:
                        continue
                    minafter = min(self.detrended_seq[i][k:lookaheadindex + 1])
                    min_af_index = np.argmin(self.detrended_seq[i][k:lookaheadindex + 1]) + k
                    if minafter < (1 - required_rise) * self.detrended_seq[i][k]:
                        self.peak_loc[i][k] = 1
                        self.peak_start[i][k] = min_bf_index
                        self.peak_half_start[i][k] = np.where(
                            self.detrended_seq[i][min_bf_index:k + 1] <= (minbefore + self.detrended_seq[i][k]) / 2)[0][
                                                         -1] + min_bf_index
                        self.peak_end[i][k] = min_af_index
                        self.peak_half_end[i][k] = np.where(
                            self.detrended_seq[i][k:min_af_index + 1] <= (minafter + self.detrended_seq[i][k]) / 2)[0][
                                                       0] + k
                        self.peak_rise_time[i][k] = k - min_bf_index
                        self.peak_fall_time[i][k] = min_af_index - k
                        height = (2 * self.detrended_seq[i][k] - minbefore - minafter) / 2
                        # height=max(self.detrended_seq[i][k]-minbefore,self.detrended_seq[i][k]-minafter)
                        pks[i].append(height)
                        self.peak_height[i][k] = height
                        prior_peak = k
            next_peak = self.length - 1
            self.peak_height_std[i] = np.std(np.array(pks[i]))
            self.peak_height_mean[i] = np.mean(np.array(pks[i]))
            continue
            for k in reversed(candidate[i]):
                lookafterindex = min(next_peak, k + lookafter)
                minafter = min(self.detrended_seq[i][k:lookafterindex + 1])
                if minafter < (1 - required_rise) * self.detrended_seq[i][k]:
                    next_peak = k
                else:
                    self.peak_loc[i][k] = 0
        for num in range(self.obs_num):
            loc = np.where(self.peak_height[num] > (max(self.detrended_seq[num]) - min(self.detrended_seq[num])) / 3)[0]
            # loc=np.where((self.peak_height[num]>self.peak_height_mean[num]+3*self.peak_height_std[num]))[0]
            self.filterer_peak_loc[num][loc] = 1
            self.filterer_peak_half_start[num][self.peak_half_start[num][loc].astype(int)] = 1
            self.filterer_peak_half_end[num][self.peak_half_end[num][loc].astype(int)] = 1
            self.filterer_peak_loc_2[num] = loc
            heights = self.peak_height[num][loc]
            rise_times = self.peak_rise_time[num][loc]
            fall_times = self.peak_fall_time[num][loc]
            self.filterer_peak_height_mean[num] = np.mean(heights)
            self.filterer_peak_height[num] = list(heights)
            self.filterer_peak_rise_time[num] = list(rise_times)
            self.filterer_peak_fall_time[num] = list(fall_times)
        for num in range(self.obs_num):
            index_lst = [1 for _ in range(self.obs_num)]
            for ind in self.filterer_peak_loc[num]:
                if ind == 1:
                    for i in range(int(self.peak_start[num][i]), int(self.peak_end[num][i] + 1)):
                        index_lst[i] = 0
            real_index = np.where(np.array(index_lst) == 1)
            other_points = self.detrended_seq[num][real_index]
            self.non_peak_std[num] = np.std(other_points)
        self.num_peak_rec = [len(self.filterer_peak_loc_2[i]) for i in range(self.obs_num)]

    def Print_Peak(self, num):
        main_data = self.detrended_seq[num]
        # loc=np.where((self.peak_height[num]>self.peak_height_mean[num]+3*self.peak_height_std[num]))[0]
        loc = np.where(self.peak_height[num] > (max(self.detrended_seq[num]) - min(self.detrended_seq[num])) / 3)[0]
        # loc=np.where(self.peak_height[num]>0)[0]
        # print(loc)
        highlight = [loc, main_data[loc]]
        plt.plot(main_data)
        plt.scatter(*highlight, marker='v', color='r')

    def Find_Peak_Good(self, thresh=0.15):
        ans = []
        for item in self.series_rel_std_sorted:
            if item[0] < thresh:
                ans.append(item[1])
        return ans

    def Find_Peak_Bood(self, thresh=0.15):
        ans = []
        for item in self.series_rel_std_sorted:
            if item[0] >= thresh:
                ans.append(item[1])
        return ans

    def Print_ALL_Peaks(self):
        path = self.filename + "_All_Peaks"
        for i in range(self.obs_num // 100 + 1):
            num_left = min(self.obs_num - 100 * i, 100)
            if num_left <= 0:
                break
            with plt.rc_context({'xtick.color': 'yellow', 'ytick.color': 'yellow'}):
                fig, axs = plt.subplots(num_left, figsize=(15, 4 * num_left))
                fig.tight_layout()
                for j in range(100 * i, 100 * i + num_left):
                    main_data = self.detrended_seq[j]
                    # loc=np.where((self.peak_height[j]>self.peak_height_mean[j]+2*self.peak_height_std[j]))[0]
                    loc = np.where(self.peak_height[j] > (max(self.detrended_seq[j]) - min(self.detrended_seq[j])) / 3)[
                        0]
                    # loc=np.where(self.peak_height[j]>0)[0]
                    highlight = [loc, main_data[loc]]
                    axs[j % 100].plot(main_data)
                    axs[j % 100].set_xlabel('Time')
                    axs[j % 100].set_ylabel('Intensity')
                    axs[j % 100].scatter(*highlight, marker='v', color='r')
                fig.savefig(path + "_All_Peaks_" + str(i))
                fig.clf()
                plt.close()

    def Raster_Plot(self):
        path = self.filename + "_Raster_Plot"
        x = []
        y = []
        for i in range(self.obs_num):
            for j in range(self.length):
                if self.filterer_peak_loc[i][j] == 1:
                    x.append(j)
                    y.append(i)
        plt.scatter(x, y, color=(0, 0.8, 0))
        plt.xlabel('Time(s)')
        plt.ylabel('ROI(#)')
        plt.show()
        plt.savefig(path)
        plt.close()

    def Histogram_Height(self):
        path = self.filename + "_Histogram_Height"
        combined = [item for sublist in self.filterer_peak_height for item in sublist]
        plt.hist(combined, bins=10, edgecolor='black', color=(0.6, 0.6, 0.75))
        plt.xlabel('height of peak')
        plt.ylabel('number of events')
        # plt.show()
        plt.savefig(path)
        plt.close()

    def Histogram_Time(self):
        path = self.filename + "_Histogram_Time"
        rise_times = [item for sublist in self.filterer_peak_rise_time for item in sublist]
        fall_times = [item for sublist in self.filterer_peak_fall_time for item in sublist]
        plt.hist([fall_times, rise_times], stacked=True, label=['fall_time', 'rise_time'],
                 color=[(0.6, 0.2, 0.2), (0.6, 0.6, 0.6)], edgecolor='black')
        plt.legend(prop={'size': 10})
        plt.xlabel('time (s)')
        plt.ylabel('number of events')
        # plt.show()
        plt.savefig(path)
        plt.close()

    # Path here needs csv extension
    def Save_Result(self):
        path1 = self.filename + "_Peak_Data.csv"
        path2 = self.filename + '_Series_Data.csv'
        path3 = self.filename + "_Summary_Data.csv"
        details = {
            'ROI(#)': [],
            'Peak_Number': [],
            'Time': [],
            'Height': [],
            'Rise_Time': [],
            'Fall_Time': [],
            'Total_Time': [],
        }
        peak_data = pd.DataFrame(details)
        for i in range(len(self.filterer_peak_height)):
            for j in range(len(self.filterer_peak_height[i])):
                peak_data.loc[len(peak_data.index)] = [int(i), int(j), self.filterer_peak_loc_2[i][j],
                                                       self.filterer_peak_height[i][j],
                                                       self.filterer_peak_rise_time[i][j],
                                                       self.filterer_peak_fall_time[i][j],
                                                       self.filterer_peak_rise_time[i][j] +
                                                       self.filterer_peak_fall_time[i][j]]
        peak_data = peak_data.astype(
            {'ROI(#)': 'int32', 'Peak_Number': 'int32', 'Time': 'int32', 'Rise_Time': 'int32', 'Fall_Time': 'int32',
             'Total_Time': 'int32'})
        peak_data.to_csv(path1, index=False)
        details = {
            'ROI(#)': [],
            'Number_of_Peaks': [],
            'Mean_Height': [],
            'Mean_Rise_Time': [],
            'Mean_Fall_Time': [],
            'Mean_Total_Time': [],
            'Mean_InterEvent_Interval': [],
            'Mean_Frequency': [],
        }
        series_data = pd.DataFrame(details)
        for i in range(len(self.filterer_peak_height)):
            interv = 0
            freq = 0
            if len(self.filterer_peak_height[i]) >= 2:
                interv = (self.filterer_peak_loc_2[i][-1] - self.filterer_peak_loc_2[i][0]) / (
                            len(self.filterer_peak_height[i]) - 1)
                freq = 1 / interv
            series_data.loc[len(series_data.index)] = [i, len(self.filterer_peak_loc_2[i]),
                                                       np.mean(self.filterer_peak_height[i]),
                                                       np.mean(self.filterer_peak_rise_time[i]),
                                                       np.mean(self.filterer_peak_fall_time[i]),
                                                       np.mean(self.filterer_peak_rise_time[i]) + np.mean(
                                                           self.filterer_peak_fall_time[i]), interv, freq]
        series_data.to_csv(path2, index=False)
        temp = np.array([max(a, 0) for a in list(series_data['Number_of_Peaks'] - 1)])
        val = np.dot(temp, np.array(series_data['Mean_InterEvent_Interval'])) / np.sum(temp)
        interlev_lst = []
        base1 = np.array(series_data['Mean_InterEvent_Interval'])
        for i in range(len(temp)):
            interlev_lst += [base1[i]] * int(temp[i])
        details = {
            'Mean_Number_of_Signal_Events': [np.mean(series_data['Number_of_Peaks'])],
            'Standard_Deviation_of_the_Number_of_Signal_Events': [np.std(series_data['Number_of_Peaks'])],
            'Mean_Height_All': [np.mean(flatten(self.filterer_peak_height))],
            'Std_Height_All': [np.std(flatten(self.filterer_peak_height))],
            'Mean_Rise_Time_All': [np.mean(flatten(self.filterer_peak_rise_time))],
            'Std_Rise_Time_All': [np.std(flatten(self.filterer_peak_rise_time))],
            'Mean_Fall_Time_All': [np.mean(flatten(self.filterer_peak_fall_time))],
            'Std_Fall_Time_All': [np.std(flatten(self.filterer_peak_fall_time))],
            'Mean_Total_Time_All': [
                np.mean(flatten(self.filterer_peak_rise_time)) + np.mean(flatten(self.filterer_peak_fall_time))],
            'Std_Total_Time_All': [
                np.std(np.add(flatten(self.filterer_peak_rise_time), flatten(self.filterer_peak_fall_time)))],
            'Mean_InterEvent_Interval_All': [val],
            'Std_InterEvent_Interval_All': [np.std(interlev_lst)],
            'Mean_Frequency_All': [1 / val],
        }
        summary_data = pd.DataFrame(details)
        summary_data.to_csv(path3, index=False)

    def Synchronization(self, cluster=False):
        path = self.filename + "_Synchronization_Plot"
        Peak_Regions = np.zeros(self.obs_num * self.length).reshape(self.obs_num, self.length)
        for i in range(self.obs_num):
            Peak_Regions[i][0] = self.filterer_peak_half_start[i][0] - self.filterer_peak_half_end[i][0]
            for j in range(1, self.length):
                Peak_Regions[i][j] = Peak_Regions[i][j - 1] + self.filterer_peak_half_start[i][j] - \
                                     self.filterer_peak_half_end[i][j]
        Peak_Regions = 2 * Peak_Regions - 1
        P = np.zeros(self.obs_num * (self.length - 1)).reshape(self.obs_num, (self.length - 1))
        for i in range(self.obs_num):
            vec = Peak_Regions[i]
            R = np.dot(vec[:, None], vec[None, :])
            for j in range(self.length - 1):
                temp_v = []
                for k in range(j, self.length):
                    temp_v.append(R[k][k - j])
                P[i][j] = np.mean(temp_v)
        SI = np.zeros(self.obs_num * self.obs_num).reshape(self.obs_num, self.obs_num)
        for i in range(self.obs_num):
            for j in range(i + 1):
                SI[i][j] = np.sum(np.dot(P[i] - np.mean(P[i]), P[j] - np.mean(P[j])) / np.std(P[i]) / np.std(P[j])) / (
                            self.length - 2)
                SI[j][i] = SI[i][j]
        SI[np.isnan(SI)] = 0
        if not cluster:
            ax = sns.heatmap(SI, cmap='jet', linecolor='black')
            ax.invert_yaxis()
            # plt.show()
            plt.savefig(path + "No_Cluster")
            plt.close()
            np.savetxt(path + "No_Cluster.csv", SI, delimiter=",")
        else:
            e_val, e_vec = LA.eig(SI)
            larg_e_val = e_val[e_val > 1]
            larg_e_vec = (e_vec.T[e_val > 1]).T
            max_index = []
            max_score = []
            if len(larg_e_val) > 0:
                ParticipationIndices = np.dot((larg_e_vec * np.abs(larg_e_vec)), np.diag(larg_e_val))
                max_index = np.argmax(ParticipationIndices, 1)
                max_score = np.max(ParticipationIndices, 1)
                max_index[np.abs(max_score) < np.finfo(float).eps] = 0
                temp = []
                for i in range(len(max_score)):
                    temp.append((max_index[i], max_score[i]))
                temp = np.array(temp, dtype="f,f")
                idx = np.argsort(temp)
                SI = SI[idx].T[idx].T
            ax = sns.heatmap(SI, cmap='jet', linecolor='black')
            ax.invert_yaxis()
            # plt.show()
            plt.savefig(path + "With_Cluster")
            np.savetxt(path + "With_Cluster.csv", SI, delimiter=",")
            plt.close()

    def Correlation(self):
        path = self.filename + "_Correlation_Plot"
        heat_mat = np.zeros(self.obs_num * self.obs_num).reshape(self.obs_num, self.obs_num)
        max_lag = min(50, self.length // 2)
        for i in range(self.obs_num):
            for j in range(i + 1):
                max_cor = -1
                for lag in range(max_lag + 1):
                    A = self.detrended_seq[i, 0:(self.length - lag)]
                    B = self.detrended_seq[j, lag:self.length]
                    try:
                        cor = np.sum((A - np.mean(A)) * (B - np.mean(B)) / np.std(A) / np.std(B)) / (
                                    self.length - lag - 1)
                    except:
                        return (A, B, i, j, lag)
                    max_cor = max(cor, max_cor)
                for lag in range(1, max_lag + 1):
                    A = self.detrended_seq[i, lag:self.length]
                    B = self.detrended_seq[j, 0:self.length - lag]
                    cor = np.sum((A - np.mean(A)) * (B - np.mean(B)) / np.std(A) / np.std(B)) / (self.length - lag - 1)
                    max_cor = max(cor, max_cor)
                heat_mat[i][j] = max_cor
                heat_mat[j][i] = max_cor
        ax = sns.heatmap(heat_mat, cmap='jet', linecolor='black', vmin=-1, vmax=1)
        ax.invert_yaxis()
        # plt.show()
        plt.savefig(path)
        np.savetxt(path + ".csv", heat_mat, delimiter=",")
        plt.close()

    def Print_Peak_Good(self, thresh=0.15):
        total = 0
        for item in self.series_rel_std_sorted:
            if item[0] < thresh:
                total += 1
        fig, axs = plt.subplots(total, figsize=(15, 300))
        fig.tight_layout()
        j = 0
        for item in self.series_rel_std_sorted:
            if item[0] < thresh:
                axs[j].plot(self.detrended_seq[item[1]])
                j += 1
        fig.savefig("/content/drive/MyDrive/Good_Ones")

    def Print_Peak_Bad(self, thresh=0.15):
        total = 0
        for item in self.series_rel_std_sorted:
            if item[0] >= thresh:
                total += 1
        fig, axs = plt.subplots(total, figsize=(15, 90))
        fig.tight_layout()
        j = 0
        for item in self.series_rel_std_sorted:
            if item[0] >= thresh:
                axs[j].plot(self.detrended_seq[item[1]])
                j += 1
        fig.savefig("/content/drive/MyDrive/Bad_Ones")

    def Filter_Series(self, model_root):
        clf = pickle.load(open(model_root, 'rb'))
        ratio = np.array(self.filterer_peak_height_mean) / np.array(self.non_peak_std)
        ratio[np.isnan(ratio)] = 1
        rel_std = self.series_rel_std
        X = np.array([[0.0] for _ in range(self.obs_num)])
        Z = np.array([0.0 for _ in range(self.obs_num)])
        for i in range(len(X)):
            X[i][0] = ratio[i]
            Z[i] = rel_std[i]
        W = np.array(self.series_mad)
        details = {
            'ROI(#)': [],
            'Number_of_Peaks': [],
            'Mean_Height': [],
            'Mean_Rise_Time': [],
            'Mean_Fall_Time': [],
            'Mean_Total_Time': [],
            'Mean_InterEvent_Interval': [],
            'Mean_Frequency': [],
        }
        series_data = pd.DataFrame(details)
        for i in range(len(self.filterer_peak_height)):
            interv = 0
            freq = 0
            if len(self.filterer_peak_height[i]) >= 2:
                interv = (self.filterer_peak_loc_2[i][-1] - self.filterer_peak_loc_2[i][0]) / (
                            len(self.filterer_peak_height[i]) - 1)
                freq = 1 / interv
            series_data.loc[len(series_data.index)] = [i, len(self.filterer_peak_loc_2[i]),
                                                       np.mean(self.filterer_peak_height[i]),
                                                       np.mean(self.filterer_peak_rise_time[i]),
                                                       np.mean(self.filterer_peak_fall_time[i]),
                                                       np.mean(self.filterer_peak_rise_time[i]) + np.mean(
                                                           self.filterer_peak_fall_time[i]), interv, freq]
        series_data = series_data.drop(columns=['ROI(#)'])
        series_data['peak_mean_prominence'] = self.peak_mean_prominence
        series_data['peak_height_std'] = self.peak_height_std
        series_data['candidate_mean_prominence'] = self.candidate_mean_prominence
        series_data['peak_std_prominence'] = self.peak_std_prominence
        series_data['W'] = W
        series_data['Z'] = Z
        series_data['X'] = X
        series_data = series_data.to_numpy()
        series_data[np.isnan(series_data)] = 0
        pred = clf.predict(series_data)
        new_idx = np.where(pred > 0)
        return self.seq[new_idx]
