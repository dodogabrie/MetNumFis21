"""
This file plot a MC history given the filename
"""

### Add to PYTHONPATH the utils folder  ############################
import os, sys
path = os.path.realpath(__file__)
main_folder = 'MetNumFis21/'
sys.path.append(path.split(main_folder)[0] + main_folder + 'utils/')
####################################################################


import numpy as np
from m1.readfile import slowload, fastload
import time 

def history(filename):





    import matplotlib.pyplot as plt

if __name__ == '__main__':
    history(filename)
