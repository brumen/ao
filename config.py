# configuration file
CUDA_PRESENT = False

import sys
import numpy as np

# adding various paths
prod_dir = '/home/brumen/work/ao/'
work_dir = prod_dir
sys.path.append(work_dir)  # basic path
subdirs = ['cubl', 'cublas', 'tests', 'cuda', 'vols', 'pricers', 'db']
for sd in subdirs:
    sys.path.append(work_dir + sd)

# cython params
cython_include_dirs = []  # '/usr/local/lib/python2.7/dist-packages/numpy/core/include/']
cython_extra_link_args = []  # '-L/usr/local/lib/python2.7/dist-packages/numpy/core/lib']

if CUDA_PRESENT:  # cuda modules
    import pycuda.gpuarray as gpa
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.curandom import rand as curand
    from pycuda.compiler import SourceModule
