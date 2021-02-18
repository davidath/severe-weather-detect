import numpy as np
import sys
import make_utils
import datetime

# Set Time frame setting - min date max date in weather data
START = datetime.datetime(1979, 1, 1, 0, 0)
END = datetime.datetime(2018, 5, 31, 18, 0)

CHANNELS = 3
REPEAT_NUM = 4
GHT = np.load(sys.argv[2])
ght = GHT['train_data']
#  noisy = make_utils.noise(ght, CHANNELS, REPEAT_NUM)
# make_utils.make_npz('notest.npz', train_data=noisy)

TEST_LEN = 17584 # ~ 3 years
# TEST_LEN = 28792

# "BIASED" TEST
# make_utils.make_tt(START, END, sys.argv[1], ght, TEST_LEN, oversample=False)
#  make_utils.make_tt(START, END, sys.argv[1], ght, TEST_LEN, oversample=True)

# Random sample TEST
#  make_utils.make_tt(START, END, sys.argv[1], ght, oversample=False)
#  make_utils.make_tt(START, END, sys.argv[1], ght, oversample=True)

# Individual weather events

#  make_utils.make_indvi(START, END, sys.argv[1], ght, 'Windstorm', oversample=True)
#  make_utils.make_indvi(START, END, sys.argv[1], ght, 'Flood', oversample=True)
#  make_utils.make_indvi(START, END, _csv, ght, 'Hail', oversample=False)
#  make_utils.make_indvi(START, END, sys.argv[1], ght, 'Tornado', oversample=True)

# ACTUAL RUNS

#  make_utils.make_tt(START, END, sys.argv[1], ght, oversample=True,
#  combine=True)
#  make_utils.make_indvi(START, END, sys.argv[1], ght, 'undersample4k',
    #  short=True, short_num=4000)
#  make_utils.make_indvi(START, END, sys.argv[1], ght, 'undersample10k',
    #  short=True, short_num=10000)
#  make_utils.make_indvi(START, END, sys.argv[1], ght, 'undersample20k',
    #  short=True, short_num=20000)
#  make_utils.make_indvi(START, END, sys.argv[1], ght, 'undersample30k',
    #  short=True, short_num=30000)

#  make_utils.make_oneall(START, END, sys.argv[1], ght, 'Windstorm', short=True,
        #  short_num=4000, anom=True)
#  make_utils.make_oneall(START, END, sys.argv[1], ght, 'Flood', short=True,
        #  short_num=4000)
#  make_utils.make_oneall(START, END, sys.argv[1], ght, 'Tornado', short=True,
        #  short_num=4000)

#  make_utils.make_allone(START, END, sys.argv[1], ght, 'Windstorm', 'Flood', short=True,
        #  short_num=500)
#  make_utils.make_allone(START, END, sys.argv[1], ght, 'Windstorm', 'Tornado', short=True,
        #  short_num=500)
#  make_utils.make_allone(START, END, sys.argv[1], ght, 'Flood', 'Windstorm', short=True,
        #  short_num=500)
#  make_utils.make_allone(START, END, sys.argv[1], ght, 'Flood', 'Tornado', short=True,
        #  short_num=500)
#  make_utils.make_allone(START, END, sys.argv[1], ght, 'Tornado', 'Windstorm', short=True,
        #  short_num=500)
#  make_utils.make_allone(START, END, sys.argv[1], ght, 'Tornado', 'Flood', short=True,
        #  short_num=500)
