import logging
import sys
import time

# Log finle enabled. global variable
# all trial will always be save to log/results_{name}.json
DEBUG = True  # save a file to log folder with what was printed at each step
VERBOSE = False  # include more informations when printing step
PRINT_ALL = False  # print all steps or just the final step per trial
PRINT_5MIN = False  # just print step log in 5 in 5 min (of market time)
PRINT_ON_TEST = True
ORDER_TO_TRACK = '000000000000000'
START_MKT_TIME = 9*60**2+20.*60
CLOSE_MKT_TIME = 15.*60**2+49.*60 + 59
STOP_MKT_TIME = 15.*60**2+35.*60  # use just by some specific agents
# number of second to the agent change its mind (learner)
WEIGHTING_TIME = 30.  # 10 or 30

# setup logging messages
s_format = '%(asctime)s;%(message)s'
s_now = time.strftime('%c')
s_now = s_now.replace('/', '').replace(' ', '_').replace(':', '')
s_file = 'log/train_test/sim_{}.log'.format(s_now)
s_log_file = s_file

root = logging.getLogger()
root.setLevel(logging.DEBUG)

formatter = logging.Formatter(s_format)

if not root.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    root.addHandler(ch)

if DEBUG:
    fh = logging.FileHandler(s_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)
