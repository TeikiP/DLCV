# -*- coding: utf-8 -*-
from __future__ import print_function
import time, random, datetime, gc
from src.functions import *
from src.model import *
from src.data import *


if __name__ == "__main__":
    make_path('/espace/DLCV')
    extract_RGB(data_path='/net/ens/DeepLearning/lab5/Data_TP/Videos', output_path='/espace/DLCV/Data')
    stat_dataset(path='/espace/DLCV/Data')
    compute_flow(data_path='/espace/DLCV/Data', flow_calculation=True)


    
