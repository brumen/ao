# front office tolling model 
# see the front office doc. for other things 

# File defines:
import config 
import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import time


# import scipy
# import scipy.optimize
# import scipy.integrate
# import scipy.special
# import scipy.stats 
# import scipy.optimize 


# CHECK WHICH OF THESE YOU REALLY NEED 
#from scipy.sparse import lil_matrix
#from scipy.sparse.linalg import spsolve
#from numpy.linalg import solve, norm
from numpy.random import rand

# import openopt (from dev. version)
# import openopt 
# import DerApproximator
# import FuncDesigner

# multiprocessing module 
#import multiprocessing


_libcublas = ctypes.cdll.LoadLibrary(config.work_dir + 'cubl/cublas_wrap.so')
#_libcublas = ctypes.CDLL('./cublas_wrap.so')
_libmagma = ctypes.cdll.LoadLibrary('/usr/local/magma-1.2/lib/libmagma.so')

def cublasIsamax_h(x_h):
    l = len(x_h)
    return _libcublas.cublasIsamax_h(x_h.ctypes.data, l)

def cublasIsamax_d(x_gpu):
    l = len(x_gpu)
    return _libcublas.cublasIsamax_d(x_gpu.ptr, l)

# vector x vector operations 
def cublasSdot_d (x_d, y_d):
    n = len (x_d)
    res = ctypes.c_float(1.)

    _libcublas.cublasSdot_d (n, x_d.ptr, y_d.ptr, ctypes.byref(res) )
    
    return res.value


# matrix x vector operations
def cublasSgemv_h (alpha, A, x, beta, y):
    A_d = gpuarray.to_gpu (A).astype(np.float32)
    x_d = gpuarray.to_gpu (x).astype(np.float32)
    y_d = gpuarray.to_gpu (y).astype(np.float32)

    cublasSgemv_d (alpha, A_d, x_d, beta, y_d)
    return y_d.get()
    

def cublasSgemv_d (alpha, A_d, x_d, beta, y_d):
    m = A_d.shape[0]
    n = A_d.shape[1]

    # THIS MIGHT BE WRONG - m,n
    _libcublas.cublasSgemv_d (n, m, ctypes.c_float (alpha), A_d.ptr, 
                              n, x_d.ptr, ctypes.c_float (beta), y_d.ptr)

    return y_d

# performs quadratic form (Ax,y)
# THIS DOES NOT WORK CORRECTLY
def cublas_quadf (A_d, x_d, y_d):

    res_d = gpuarray.zeros ( len(y_d), np.float32 )
    cublasSgemv_d (1.0, A_d, x_d, 0., res_d)
    return gpuarray.dot (res_d, y_d).get() + 0.



# matrix x matrix operations (C is useless), 
# but _HAS_ to be there 
def cublasSgemm_h (alpha, A, B, C):
    m = A.shape[0] # rows of A
    n = A.shape[1] # cols of A, rows of B
    k = B.shape[1] # cols of B 

    A_d = gpuarray.to_gpu ( A ) # A_d is shape n x m
    B_d = gpuarray.to_gpu ( B ) # B_d is shape k x n
    C_d = gpuarray.to_gpu ( C ) # C_d is shape k x m

    _libcublas.cublasSgemm_d (m, k, n, ctypes.c_float (alpha), A_d.ptr, n,
                              B_d.ptr, k, ctypes.c_float (0.), C_d.ptr, m)


    # transformation from row-major to column major
    return  (np.ravel (C_d.get()).reshape (k,m)).transpose()


# LU decomposition of A_d, new matrix returned
def magma_sgetrf_d (A_d):
    A_s = A_d.shape
    magma_lu = _libmagma.magma_sgetrf_gpu

    # info = 0, means that the routine finshed successfully 
    # piv (pivot vector), does _not_ work, but is needed 
    info = ctypes.c_int(1)
    info_p = ctypes.pointer(info)
    piv = np.arange(min(A_s[0], A_s[1])).astype(np.int)

    A_d = gpuarray.to_gpu ( np.reshape(A_h.astype(np.float32), 
                                       A_s[0] * A_s[1], order='F') )
    ret = magma_lu (A_s[0], A_s[1], A_d.ptr, A_s[0], piv.ctypes.data, info_p)
    
    return np.reshape (A_d.get(), (A_s[0], A_s[1]), order='F')

# LU where A_h is on the host 
def magma_sgetrf_h (A_h):
    A_s = A_h.shape
    magma_lu = _libmagma.magma_sgetrf

    # info = 0, means that the routine finshed successfully 
    # piv (pivot vector), does _not_ work, but is needed 
    info = ctypes.c_int(1)
    info_p = ctypes.pointer(info)
    piv = np.arange(min(A_s[0], A_s[1])).astype(np.int)

    A_n = np.reshape(A_h.astype(np.float32), A_s[0] * A_s[1], order='F')
    ret = magma_lu (A_s[0], A_s[1], A_n.ctypes.data, A_s[0], piv.ctypes.data, info_p)
    
    return np.reshape (A_n, (A_s[0], A_s[1]), order='F')

# NOT FINISHED NOT FINISHED
# qr factorization of A_h 
def magma_sgeqrf_h (A_h):
    A_s = A_h.shape
    sgeqrf = _libmagma.magma_sgeqrf
    m = A_s[0]
    n = A_s[1]
    lda = m
    nb = _libmagma.magma_get_sgeqrf_nb (m)
    
    tau = np.zeros(min(n,m)) 
    lwork = n * nb 
    work = np.zeros (lwork)

    # b2 = 0, means that the routine finshed successfully 
    # piv (pivot vector), does _not_ work, but is needed 
    info = ctypes.c_int(1)
    info_p = ctypes.pointer(info)

    A_n = np.reshape(A_h.astype(np.float32), m*n, order='F')
    ret = sgeqrf (m, n, A_n.ctypes.data, lda, tau.ctypes.data, work.ctypes.data,
                  lwork, info_p)
    
    return np.reshape (A_n, (m,n), order='F')






# General linear equation solver
# A _has_ to be squared 
def magma_sgetrs_h (A_h, b_h):
    magma_sgetrs = _libmagma.magma_sgetrs_gpu
    A_s = A_h.shape
    b_s = b_h.shape

    trans = ctypes.c_char ('N')
    nrhs = b_s[1]
    lda = A_s[0]
    ldb = b_s[0]
    N = A_s[1]
    
    print trans
    print nrhs
    print lda
    print ldb
    print N

    #A_n = magma_sgetrf_h (A_h)
    print A_h
    print b_h
    #print A_n

    # b2 = 0, means that the routine finshed successfully 
    # piv (pivot vector), does _not_ work, but is needed 
    b2 = ctypes.c_int(1)
    b1 = ctypes.pointer(b2)
    piv = np.arange(N).astype(np.int)
    print piv 
    A_d = gpuarray.to_gpu ( np.reshape(A_h.astype(np.float32), 
                                       A_s[0] * A_s[1], order='F') )
    b_d = gpuarray.to_gpu ( np.reshape(b_h.astype(np.float32), 
                                       b_s[0] * b_s[1], order='F') )

    #h_w = np.zeros( (N,nrhs), dtype=np.float32)
    # print h_w
    print piv
    #ret = magma_sgetrs (trans, N, nrhs, A_d.ptr, lda, piv.ctypes.data, b_d.ptr, ldb, b1)
    
    ret = magma_sgetrs (trans, N, nrhs, 
                        A_d.ptr, lda, piv.ctypes.data, 
                        b_d.ptr, ldb, b1)
    # h_w.ctypes.data )

    return np.reshape (b_d.get(), (b_s[0], b_s[1]), order='F')


# General linear equation solver
# A _has_ to be squared 
def magma_sgesv_h (A_h, b_h):
    magma_sgesv = _libmagma.magma_sgesv
    A_s = A_h.shape
    b_s = b_h.shape

    nrhs = b_s[1]
    lda = A_s[0]
    ldb = b_s[0]
    N = A_s[1]
    
    # info = 0, means that the routine finshed successfully 
    # piv (pivot vector), does _not_ work, but is needed 
    info = ctypes.c_int(1)
    info_p = ctypes.pointer(info)
    piv = np.arange(N).astype(np.int)

    A_n = np.reshape(A_h.astype(np.float32), A_s[0] * A_s[1], order='F')
    b_n = np.reshape(b_h.astype(np.float32), b_s[0] * b_s[1], order='F')

    ret = magma_sgesv (N, nrhs, 
                       A_n.ctypes.data, lda, piv.ctypes.data, 
                       b_n.ctypes.data, ldb, info_p)

    return np.reshape (b_n, (b_s[0], b_s[1]), order='F')
    
