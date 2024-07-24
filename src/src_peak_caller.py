"""
Source file for Section 4 - Peak Caller
Last edited on Jun 14, 2023
Author: Xuchen Wang (xw2747@columbia.edu), Yian Wang (canying0913@gmail.com)
For all inquiry, please contact Xuchen Wang.
Copyright Yian Wang - 2023
"""
import pickle
from src.utils import iprint
from src.code.time_series_clustering import ts_clustering
from src.code.time_lapse_detect import tl_detect
from src.code.test import stat_test
from src.code.peakcaller import PeakCaller
from src.code.phase_analysis import analysis


class PC:
    def __init__(self):
        pass

    def log(self, msg: str):
        iprint(msg, log_queue=self.log_queue)

    def run(self):
        ts_clustering()
        tl_detect()
        stat_test()
        self.peakcaller.run()
        analysis()
