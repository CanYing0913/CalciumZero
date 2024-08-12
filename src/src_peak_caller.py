"""
Source file for Section 4 - Peak Caller
Last edited on Jun 14, 2023
Author: Xuchen Wang (xw2747@columbia.edu), Yian Wang (canying0913@gmail.com)
For all inquiry, please contact Xuchen Wang.
Copyright Yian Wang - 2023
"""
from pathlib import Path
from src.utils import iprint
from src.code.time_series_clustering import ts_clustering
from src.code.time_lapse_detect import tl_detect
from src.code.stat_test import stat_test
from src.code.peakcaller import PeakCaller
from src.code.phase_analysis import analysis

import numpy as np


class PC:
    def __init__(self, save_dir, log_queue=None):
        assert Path(save_dir).exists(), f'{save_dir} does not exist!'
        self.save_dir = save_dir
        self.log_queue = log_queue
        Path(save_dir).joinpath('data').joinpath('phase').mkdir(exist_ok=True, parents=False)
        plot = Path(save_dir).joinpath('plot')
        plot.joinpath('time_lapse_plot').joinpath('cluster').mkdir(parents=True, exist_ok=True)
        plot.joinpath('time_lapse_plot').joinpath('scale_cluster').mkdir(parents=True, exist_ok=True)
        Path(save_dir).joinpath('test').mkdir(exist_ok=True, parents=False)
        Path(save_dir).joinpath('peakcaller').mkdir(exist_ok=True, parents=False)
        self.peakcaller = PeakCaller(seq=None, save_base=str(Path(save_dir).joinpath('peakcaller')))

    def log(self, msg: str):
        iprint(msg, log_queue=self.log_queue)

    def run(self, filename):
        ts_clustering(filename, self.save_dir)
        tl_detect(self.save_dir)
        stat_test(self.save_dir)
        seq = np.load(filename)
        self.peakcaller.run(seq=seq)
        analysis(self.save_dir)
