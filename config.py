# configuration file
CUDA_PRESENT = False

import sys

# Adding paths
prod_dir = '/home/brumen/work/ao/'
work_dir = prod_dir
tmp_dir  = '/tmp/ao/'
sys.path.append(work_dir)  # basic path
subdirs = ['cubl', 'cublas', 'tests', 'cuda', 'vols', 'pricers', 'db']
for sd in subdirs:
    sys.path.append(work_dir + sd)

# cython params
cython_include_dirs = []  # '/usr/local/lib/python2.7/dist-packages/numpy/core/include/']
cython_extra_link_args = []  # '-L/usr/local/lib/python2.7/dist-packages/numpy/core/lib']
