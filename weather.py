#
# File defines:
#   mrd skew model for commodities (state reference)
#   a general diffusion model

# import config 

import numpy as np
import openopt # optimization solver
import pycuda.curandom
import pycuda.gpuarray as gpa
import pycuda.cumath
import pycuda.reduction
from pycuda.compiler import SourceModule
#from cudanormal import cudanormal
import datetime
import calendar

#import cublas


# reduction kernel, sums the elemnts in a vector 
average_reduction = pycuda.reduction.ReductionKernel(np.dtype(np.float32), neutral="0",
                                                     reduce_expr="a+b", 
                                                     map_expr="x[i]",
                                                     arguments="float *x")


# vector + matrix slicing kernel - vpm
# vector * matrix slicing kernel - vtm
# TO CORRECT: N_STEP IS FIXED. 

vtpm_code = """
__global__ void vpm (float *v, float *m, int nb_cols, int nb_rows, int to_do_rows ) {
  int ind1, res_idx;
  int th_idx = threadIdx.x;
  int th_bl_idx = th_idx + blockIdx.x * blockDim.x;
  __shared__ float v_cache[31];
  v_cache[th_idx] = v[th_idx];

  for (ind1 = 0; ind1 < (to_do_rows); ind1 = ind1 + 1) {
    res_idx = ind1 * (nb_cols) * 65535 + th_bl_idx;
    if ( res_idx < ( nb_rows ) * (nb_cols) )
      m[res_idx] += v_cache[th_idx];
  }
}

__global__ void vtm (float *v, float *m, int nb_cols, int nb_rows, int to_do_rows ) {
  int ind1, res_idx;
  int th_idx = threadIdx.x;
  int th_bl_idx = th_idx + blockIdx.x * blockDim.x;
  __shared__ float v_cache[31];
  v_cache[th_idx] = v[th_idx];

  for (ind1 = 0; ind1 < (to_do_rows); ind1 += 1) {
    res_idx = ind1 * (nb_cols) * 65535 + th_bl_idx;
    if ( res_idx < (nb_rows) * (nb_cols) )
      m[res_idx] *= v_cache[th_idx];
  }
}
"""

# same as above, except that the multiplication is on cols
vtpm_cols_code = """
__global__ void vpm_cols (float *v, float *m, int nb_cols ) {

  int row_idx = blockIdx.y; 
  int col_idx = threadIdx.x + blockIdx.x * blockDim.x;

  float v_curr = v[row_idx];

  //if (threadIdx.x == 0)
  //  v_curr = v[row_idx]; 

  if ( col_idx < nb_cols )
    m[row_idx * nb_cols + col_idx ] += v_curr;

}

__global__ void vtm_cols (float *v, float *m, int nb_cols) {

  int row_idx = blockIdx.y; 
  int col_idx = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ float v_curr;

  if (threadIdx.x == 0)
    v_curr = v[row_idx];
 
  //if (threadIdx.x == 0)
  //  v_curr = v[row_idx]; 

  if ( col_idx < nb_cols )
    m[row_idx * nb_cols + col_idx ] *= v_curr;

}

__global__ void vtm_cols2 ( float v, float *m, int nb_cols, int row_idx ) {

  //int row_idx = blockIdx.y; 
  int col_idx = threadIdx.x + blockIdx.x * blockDim.x;

  //float v_curr = v[row_idx]; 
  
  //if (threadIdx.x == 0)
  //  v_curr = v[row_idx]; 

  if ( col_idx < nb_cols )
    m[row_idx * nb_cols + col_idx ] *= v;

}

__global__ void vpm_cols2 ( float v, float *m, int nb_cols, int row_idx ) {

  //int row_idx = blockIdx.y; 
  int col_idx = threadIdx.x + blockIdx.x * blockDim.x;

  //float v_curr = v[row_idx]; 
  
  //if (threadIdx.x == 0)
  //  v_curr = v[row_idx]; 

  if ( col_idx < nb_cols )
    m[row_idx * nb_cols + col_idx ] += v;

}


"""




# vector + matrix function on rows 
vtpm_module = SourceModule(vtpm_code)
vpm_f = vtpm_module.get_function ("vpm")
vtm_f = vtpm_module.get_function ("vtm")

# vector + matrix function on columns 
vtpm_cols_module = SourceModule(vtpm_cols_code)
vpm_cols_f = vtpm_cols_module.get_function ("vpm_cols")
vtm_cols_f = vtpm_cols_module.get_function ("vtm_cols")
vpm_cols2_f = vtpm_cols_module.get_function ("vpm_cols2")
vtm_cols2_f = vtpm_cols_module.get_function ("vtm_cols2")

def vtpm(v,m, tm_ind = 'p'):
    """ 
    vector times matrix, by rows 
    tm_ind = p ... for summation (plus)
    tm_ind = t ... for multiplication (times)
    """
    m_cols = m.shape[1]
    m_rows = m.shape[0]
    nb_launches = m_rows / 65535 +1 # this is an integer

    block_dims = (m_cols,1,1)
    vtpm_f = {'t': vtm_f, 'p': vpm_f} # for a single launch this works best

    if m_rows / 65535 > 0: 
        grid_dims = (65535, 1)
        vtpm_f[tm_ind](v,m, np.int32(m_cols), np.int32(m_rows), np.int32(nb_launches), 
                       block=block_dims, grid= grid_dims)
    else:
        grid_dims = (m_rows, 1)
        vtpm_f[tm_ind]( v, m, np.int32(m_cols), np.int32(m_rows), np.int32(1),
                         block = block_dims, grid = (grid_dims) )

def vtpm_cols( v, m, tm_ind = 'p'):
    """ 
    v ... vector on host
    m ... matrix on the device 
    vector times matrix, by columns 
    tm_ind = p ... for summation (plus)
    tm_ind = t ... for multiplication (times)
    """
    m_cols = m.shape[1]
    m_rows = m.shape[0]
    #nb_launches = m_rows / 65535 +1 # this is an integer

    #grid_dims = (m_cols / 512 + 1, m_rows)
    grid_dims = (m_cols / 512 + 1, 1)
    block_dims = (512,1,1)
    vtpm_cols_f = {'t': vtm_cols2_f, 'p': vpm_cols2_f} # for a single launch this works best

    # if m_cols / 65535 > 0: 
    #     grid_dims = (65535, 1)
    #     vtpm_f[tm_ind](v,m, np.int32(m_cols), np.int32(m_rows), np.int32(nb_launches), 
    #                    block=block_dims, grid= grid_dims)
    # else:
    #     grid_dims = (m_rows, 1)
    #     vtpm_f[tm_ind]( v, m, np.int32(m_cols), np.int32(m_rows), np.int32(1),
    #                      block = block_dims, grid = (grid_dims) )

    #if tm_ind == 'p':
    #vtpm_cols_f[tm_ind]( v, m, np.int32(m_cols), block=block_dims, grid=grid_dims)

    for i in range(m_rows):
        vtpm_cols_f[tm_ind] ( np.float32(v[i]), m, np.int32(m_cols), np.int32(i), 
                              block=block_dims,
                              grid=grid_dims
                              )



        

# computes a**v, where a is a number, v is a vector 
# --- became part of gpuarray - apow function 
#vpow_module = SourceModule(vpow_code)
#vpow_f = vpow_module.get_function ("vpow") # vector + matrix function 
#vpow = lambda a,v: vpow_f(a,v, block=(1,1,1), grid=(len(v),1))


cumsum_cuda_code = """
__global__ void cumsum_cuda (float *M, int nb_cols, int nb_rows, int to_do_rows) {

  int ind1, ind2, curr_row;
  float curr_val;
  int row_start_idx;

  for (ind1 = 0; ind1 < (to_do_rows); ind1 += 1) { /* traversing rows */
    row_start_idx = ind1 * nb_cols * 65535 + blockIdx.x * nb_cols;
    curr_row = ind1 * 65535 + blockIdx.x;
    if ( curr_row < nb_rows ) {
      curr_val =  M[row_start_idx]; 
      for (ind2 = 1; ind2 < nb_cols; ind2 += 1) { /* traversing individual row across cols */
        curr_val = curr_val + M[row_start_idx + ind2];
        M[row_start_idx + ind2] = curr_val;
      }
    }
  }
}
"""
cs_module = SourceModule(cumsum_cuda_code)
cs_cuda_f = cs_module.get_function ("cumsum_cuda") 

# row cumsum function (on cuda, m is a matrix), replaces m_d with 
def cumsum_cuda(m_d):

    m_cols = m_d.shape[1]
    m_rows = m_d.shape[0]
    nb_launches = m_rows / 65535 +1 # this is an integer

    block_dims = (1,1,1)
    grid_dims = (65535, 1) # rows
    cs_cuda_f(m_d, np.int32(m_cols), np.int32(m_rows), np.int32(nb_launches), 
              block=block_dims, grid= grid_dims)

    return 0


# maximum code
# maximum function, does max (M,0) by elements 
maximum_cuda_code = """
__global__ void maximum_cuda ( float *M, int nb_cols, int nb_rows ) {

  //int row_idx = blockIdx.y; 
  int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int elt_idx = blockIdx.y * nb_cols + col_idx;

  if ( col_idx < nb_cols )
      M[elt_idx] = max (M[elt_idx], 0.0);

}
"""


# maximum_cuda, computes max (M,0), overwrites M
maximum_cuda_module = SourceModule(maximum_cuda_code)
maximum_cuda_f = maximum_cuda_module.get_function ("maximum_cuda") 

def maximum_cuda (m_d):
    
    if len(m_d.shape) == 1:
        m_rows = 1
        m_cols = m_d.shape[0]
    else:
        m_rows = m_d.shape[0]
        m_cols = m_d.shape[1]

    block_dims = (512,1,1)
    grid_dims = (m_cols/512 + 1, m_rows)

    maximum_cuda_f(m_d, np.int32(m_cols), np.int32(m_rows), 
                   block=block_dims, grid= grid_dims)



# write the cumsum of M, the last column in v 
# rows is large, cols is much smaller 
rowsum_cuda_code_backup = """
__global__ void rowsum_cuda (float *M, float *v, 
                             int nb_cols, int nb_rows, int to_do_rows) {

  int ind1, ind2, curr_row;
  float curr_val;
  int row_start_idx;
  // __shared__ float curr_row[31];
  
  for (ind1 = 0; ind1 < to_do_rows; ind1 += 1) { /* traversing rows */
    row_start_idx = ind1 * nb_cols * 65535 + blockIdx.x * nb_cols;
    curr_row = ind1 * 65535 + blockIdx.x;
    if ( curr_row < nb_rows ) {
      curr_val =  M[row_start_idx]; 
      for (ind2 = 1; ind2 < nb_cols; ind2 += 1) { /* traversing individual row across cols */
        curr_val = curr_val + M[row_start_idx + ind2];

      }
    v[curr_row] = curr_val;
    }
  }
}
"""




rowsum_cuda_code = """
__global__ void rowsum_cuda (float *M, float *v, 
                             int nb_cols, int nb_rows, int to_do_rows) {

  int ind1, ind2;
  int row_start_idx;
  int curr_row = blockIdx.x * blockDim.x + threadIdx.x;

  // __shared__ float curr_col[];

  /* initial step */ 
  if (curr_row < nb_rows )
    v[curr_row] = M[curr_row]; 

  /* for each column */
  for (ind2 = 1; ind2 < nb_cols; ind2 += 1)
    if ( curr_row < nb_rows ) 
      v[curr_row] +=  M[ind2 * nb_rows + curr_row]; 

  /* final assignement */
  //if (curr_row < nb_rows )
  //  v[curr_row] = 1.0; // curr_col[curr_row];

}
"""


# cumsum over rows 
#
colsum_cuda_code = """
__global__ void colsum_cuda (float *M, int nb_cols, int nb_rows, int row_idx) {

  int col_idx = threadIdx.x + blockIdx.x * blockDim.x;

  if ( col_idx < nb_cols )
    M[row_idx * nb_cols + col_idx ] += M[(row_idx-1)*nb_cols + col_idx];

}

__global__ void colsum_cuda_last (float *M, float *res, int nb_cols, int nb_rows, int row_idx) {

  int col_idx = threadIdx.x + blockIdx.x * blockDim.x;

  if ( col_idx < nb_cols )
    res[ col_idx ] += M[row_idx*nb_cols + col_idx];

}


"""



rs_module = SourceModule(rowsum_cuda_code)
rs_cuda_f = rs_module.get_function ("rowsum_cuda") 
cs_module = SourceModule(colsum_cuda_code)
cs_cuda_f = cs_module.get_function ("colsum_cuda")
cs_cuda_last_f = cs_module.get_function ("colsum_cuda_last")

def colsum_cuda (m_d):

    m_cols = m_d.shape[1]
    m_rows = m_d.shape[0]
    #nb_launches = m_rows / 65535 +1 # this is an integer

    block_dims = (512,1,1)
    grid_dims = (m_cols/512 + 1, 1) 
    for i in range(1,m_rows):
        cs_cuda_f(m_d, np.int32(m_cols), np.int32(m_rows), np.int32(i), 
                  block=block_dims, grid= grid_dims)

def colsum_cuda_last (m_d):

    m_cols = m_d.shape[1]
    m_rows = m_d.shape[0]

    res_d = gpa.zeros ( m_cols, np.float32)

    block_dims = (512,1,1)
    grid_dims = (m_cols/512 + 1, 1) 
    for i in range(m_rows):
        cs_cuda_last_f(m_d, res_d, np.int32(m_cols), np.int32(m_rows), np.int32(i), 
                  block=block_dims, grid= grid_dims)

    return res_d


# row sum cuda - same as cumsum cuda, just that it returns last col of the matrix, 
# a final cummulative sum 
def rowsum_cuda (m_d, ones_d, rs_res_d):

    # m_cols = m_d.shape[1]
    # m_rows = m_d.shape[0]
    # ones_d = gpa.to_gpu ( np.ones ((m_cols,1)).astype(np.float32) )
    cublas.cublasSgemv_d (1., m_d, ones_d, 0., rs_res_d)


    #return rs_res_d


def rowsum_cuda_backup(m_d):

    rs_res_d = gpa.zeros ( m_d.shape[0], np.float32) # row-sum result on device 
    m_cols = m_d.shape[1]
    m_rows = m_d.shape[0]
    nb_launches = m_rows / 65535 +1 # this is an integer

    block_dims = (1,1,1)
    grid_dims = (65535, 1) # rows
    rs_cuda_f(m_d, rs_res_d, np.int32(m_cols), np.int32(m_rows), np.int32(nb_launches), 
              block=block_dims, grid= grid_dims)
    
    return rs_res_d



def rowsum_cuda_notransfer(m_d, rs_res_d):
    """ 
    sums of rows of the matrix m_d are written in v_d, which is already a 
    device vector 
    """
    # size rs_res_d == m_rows

    m_cols = m_d.shape[1]
    m_rows = m_d.shape[0]
    nb_launches = m_rows / 65535 +1  # this is an integer

    block_dims = (1,1,1)
    grid_dims = (65535, 1) # rows
    rs_cuda_f(m_d, rs_res_d, np.int32(m_cols), np.int32(m_rows), np.int32(nb_launches), 
              block=block_dims, grid= grid_dims)
    
    return rs_res_d


# BACKUP COPY, DO NOT TOUCH
def rowsum_cuda_notransfer_backup(m_d, rs_res_d):
    """ 
    sums of rows of the matrix m_d are written in v_d, which is already a 
    device vector 
    """
    # size rs_res_d == m_rows

    m_cols = m_d.shape[1]
    m_rows = m_d.shape[0]
    nb_launches = m_rows / 65535 +1  # this is an integer

    block_dims = (1,1,1)
    grid_dims = (65535, 1) # rows
    rs_cuda_f(m_d, rs_res_d, np.int32(m_cols), np.int32(m_rows), np.int32(nb_launches), 
              block=block_dims, grid= grid_dims)
    
    return rs_res_d


sin_cos_fast_code = """
__global__ void sin_fast(float *x, float *y, int x_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_size) 
      y[idx] = sinf (x[idx]);
}

__global__ void cos_fast(float *x, float *y, int x_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_size) 
      y[idx] = cosf (x[idx]);
}
"""

sin_cos_fast_module = SourceModule(sin_cos_fast_code)
sin_cos_f = {'sin': sin_cos_fast_module.get_function ("sin_fast"),
             'cos': sin_cos_fast_module.get_function ("cos_fast")}

def sin_cos_d ( x, y, sin_cos = 'sin'):
    """
    implements the sin, cos on x, and writes it in y
    """
    x_len = len (x)
    sin_cos_f[sin_cos](x, y, np.int32(x_len), block=(512,1,1), grid=(x_len/512 + 1,1) )


# writes the vector v in col n of matrix m 
# nb_sims is the number of rows (simulations in rows)
# nb_cols ... number of columns 
write_vec_in_mat_col_code = """
__global__ void write_vec_in_mat_col (float *v, float *m, int n, int nb_sims, int nb_cols) {
  
  int elt_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (elt_idx < nb_sims ) 
    m[ n  + elt_idx * nb_cols ] = v[elt_idx];
}

__global__ void write_vec_in_mat_row (float *v, float *m, int n, int nb_sims, int nb_cols) {
  
  int elt_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (elt_idx < nb_sims ) 
    m[ n  + elt_idx * nb_cols ] = v[elt_idx];
}
"""
wohdd_module = SourceModule(write_vec_in_mat_col_code)
wohdd_f = wohdd_module.get_function ("write_vec_in_mat_col") 
def write_vec_in_mat_col ( rowsum_vec_d, hdd_sim_d, n):
    """
    implements the following: hdd_sim[n,:] = rowsum_vec_d
    """

    nb_sims = hdd_sim_d.shape[0] # nb. rows, simulations in rows 
    nb_days = hdd_sim_d.shape[1] # nb. cols, days are in cols

    wohdd_f (rowsum_vec_d, hdd_sim_d, np.int32(n), np.int32(nb_sims), np.int32(nb_days),
             block=(nb_sims / 65535 +1,1,1), grid= (65535,1) )



class weather():

    def __init__(self, sim_size, hp, date_l, sp=np.array([3.3, 3.3]), 
                 gpu_ind = False):
        """
        sim_size ... simulation size 
        hp ... historical parameters 
        sp ... simulation parameters
        date_l ... date list of [date_o, date_p, date_s]
        """
        self.sim_size = sim_size
        self.N_step = 31
        self.gpu_ind = gpu_ind # True for CUDA, False for CPU 

        self.hp = hp # historical parameters 
        self.sp = sp # simulation parameters 
        self.date_o, self.date_p, self.date_s = date_l # origin, pricing, start date 

        self.T_help_d = gpa.zeros ( self.N_step, np.float32)
        self.T_d_help_d = gpa.zeros (self.N_step, np.float32)
        
        self.Z_m = None
        self.Z_m_d = None 
        self.update_sim_nb(sim_size) # constuct Z_m, Z_m_d 

    def update_sim_nb(self, sim_size):
        self.Z_m = np.random.normal(size=(31, sim_size))
        self.Z_m_d = gpa.to_gpu (self.Z_m.astype(np.float32))

    # average function
    def T_m (self, t):
        A, B, C, omega, phi, a = self.hp # a is the mean reversion params
        return A + B * t + C * np.sin (omega * t + phi)

    # average function for t_d on the device
    def T_m_d (self, t_d):
        A, B, C, omega, phi, a = self.hp # a is the mean reversion params
        t_d1 = t_d *omega + phi
        sin_cos_d(t_d1, self.T_help_d)
        return A + B * t_d + C * self.T_help_d

    def T_m_real (self, t_date):
        t_num = (t_date - self.date_o).days / 365.25
        return self.T_m (t_num)

    def T_m_der (self, t):
        A, B, C, omega, phi, a = self.hp
        return B + C * np.cos (omega * t + phi) * omega

    # same as above function, it just does the t_d on the device
    def T_m_der_d (self, t_d):
        A, B, C, omega, phi, a = self.hp
        # implements: return B + C * pycuda.cumath.cos (omega * t_d + phi) * omega
        sin_cos_d(omega * t_d + phi, self.T_d_help_d, 'cos')
        return B + ( C * omega ) * self.T_d_help_d


    def T_m_der_real (self, t_date, t_origin):
        t_num = (t_date - t_origin).days / 365.25
        return self.T_m_der (t_num)


    def T_step (self, t, dt, T_t_v, Z):
        """
        step of simulation 
          sp ... simulation parameters (sigma, lam) 
          hp ... historical parameters, see T_m
          Z is a vector of same len. as T_t_v
        """

        T_m_v = self.T_m (t)
        T_m_d = self.T_m_der (t)
        a = self.hp[5]
        sigma, lam = self.sp
        return T_t_v + (T_m_d + a * (T_m_v - T_t_v ) - lam * sigma ) * dt + sigma * np.sqrt (dt) * Z

    # complete simulates the whole process, slow 
    def T_sim (self, t_0, t_step, T_0 ):
        T_0_v = T_0 * np.ones(self.Z_m.shape[0])
        T_s = np.zeros ((self.Z_m.shape[0], self.Z_m.shape[1]+1)) # simulated matrix (
        T_s[:,0] = T_0_v
        for n in np.arange(1,self.N_step+1):
            T_s[:,n] = self.T_step(t_step * n, t_step, T_s[:,n-1], self.Z_m[:,n-1])
    
        return T_s

    def HDD (self, t_0, t_step, T_0, sp):
        ttdp = (self.date_p - self.date_o).days / 365.25
        t_v = t_0 + t_step * np.arange(self.N_step) 
        T_m_v = self.T_m (ttdp + t_v )
        T_m_d1 = self.T_m_der (ttdp + t_v) # vector as well

        T_sm = self.T_sim_inn ( T_m_v, T_m_d1, t_step, sp ) # T simulated matrix 

        return self.HDD_payoff(T_sm)


    def month_into_sigma(self, n, mi_dec):
        """
        maps month n = 0, 1, 2, 3 into sp_l index when mi_dec is given 
        """
        np1 = n+1
        return sum ([ m[1]<=np1 for m in mi_dec.values() ])

    def HDD_real (self, nb_months, sp_l, HDD_date_l ):
        """
        incorporates correct handling of dates 
        sp_l, hp_l are lists of simulation parameters, historical parameters for months 
        date_p ... pricing date (datetime format)
        date_s ... start date, 
        date_o ... origin date (check what that really is)
        nb. months ... number of months of the HDD
        """

        mi = self.months_index (HDD_date_l, sp_l[0]) # just need month_decom 
        mi_dec = self.month_decomp (mi["month_decomp"])

        hdd_val = 0.
        for n in np.arange (nb_months):
            t_month_start = self.add_months (self.date_s, n) 
            t_month_start_num = (t_month_start - self.date_p).days / 365.
        
            T_0 = self.T_m_real (t_month_start)
            t_step = 1. / 365.
            # N_step = (self.add_months(t_month_start, 1) - t_month_start).days 
            # CHECK THIS MIGHT BE WRONG 
            sp = sp_l[self.month_into_sigma(n,mi_dec)] # mapping into sp_l[ n
            if self.gpu_ind == False:
                hdd_val += self.HDD( t_month_start_num, t_step, T_0, sp)
            else:
                # 31 IS WRONG, due to impossible slicing in gpuarray
                hdd_val += self.HDD_d( t_month_start_num, t_step, T_0, sp )

        return hdd_val

    def HDD_histo (self, nb_months, HDD_date_l):

        sp_l = [(0.,0.)] * len(HDD_date_l)
        return self.HDD_real ( nb_months, sp_l, HDD_date_l )


    def month_decomp (self, hdd_l):
        """
        months decomposition 
        hdd ... hdd list is in the form (a, a_1), (a,a_2), ... 
        same start for all 
        this is used to determine the volatility structure by months 
        """

        hdd_start = hdd_l[0][0] # start is the same for all lists 
        hdd_l_sorted = sorted(hdd_l, key=lambda hdd: hdd[1])  # sorted hdd_l
        hdd_overlap_l = {0: (hdd_start, hdd_l_sorted[0][1])} # first element
        last_ind = hdd_l_sorted[0][1]
        for e in enumerate(hdd_l_sorted[1:]):
            hdd_overlap_l[e[0]+1] = (last_ind, e[1][1]) # first index was already handled
            last_ind = e[1][1]
        return hdd_overlap_l


    def add_months(self, sourcedate, months):
        """
        adds months to sourcedate (in datetime format)
        """
        month = sourcedate.month - 1 + months
        year = sourcedate.year + month / 12
        month = month % 12 + 1
        day = min(sourcedate.day,calendar.monthrange(year,month)[1])
        return datetime.date(year,month,day)

    def months_index(self, HDD_date_l, sl_init ):
        """
        construct months index 
        HDD date_l is in the dictionary form {k: (date_s, nb_months)}
        sl_init ... initial value for (sigma, lambda)
        """

        months_input = [ (1, k+1) for x,k in HDD_date_l.values() ]
        sigma_lam_l = [ sl_init for m in months_input ]
        return {"month_decomp": months_input, 
                "sigma_lam": sigma_lam_l
                }
        

    def HD_calib_all(self, HDD_date_l, HDD_price_l, HDDO_price_l):
        """
        calibrates everything 
        """
        mi = self.months_index (HDD_date_l, self.sp) # mi ... month index 
        # mi_dec = month_decomp (mi["month_decomp"])
    
        sl_calib = [self.sp] * len(HDD_date_l)

        for nb_idx in np.arange(len(self.sp)): # go over all contracts 
            # do the calibration of sigma_lambda 
            print "calibrating ", nb_idx
            def opt_fct (sl):
                sl_calib[nb_idx] = sl
                nb_months = mi["month_decomp"][nb_idx][1]-1
                hdd1 = (self.HDD_real(nb_months, sl_calib, HDD_date_l) - HDD_price_l[nb_idx])**2
                hdd2 = (self.HDDO_real (HDDO_price_l[nb_idx][0], nb_months, sl_calib, HDD_date_l ) - HDDO_price_l[nb_idx][1])**2

                return hdd1 + hdd2

            p = openopt.NLP ( opt_fct, self.sp, lb=[0., - np.inf ] )
            sl = p.solve ('scipy_cobyla').xf
            sl_calib[nb_idx] = sl

        # self.sp = sl_calib

        return sl_calib

    def HDD_payoff(self, T_sm):
        return np.average (np.sum (np.maximum (65. - T_sm , 0.), axis = 0))

    # same as above, just computes a bunch of stuff on the device 
    def HDD_payoff_d (self, T_sm_d):
        T_sm_tmp_d = 65. - T_sm_d
        maximum_cuda ( T_sm_tmp_d ) # this changes T_sm_d, which souldnt
        # rowsum_cuda_notransfer(T_sm_tmp_d, rowsum_vec_d) # writes in rowsum_vec_d
        rowsum_vec_d = colsum_cuda_last(T_sm_tmp_d)

        # the reason why this is better is that it only carries over one number
        # implements return np.average( rowsum_vec_d.get() )    
        return average_reduction(rowsum_vec_d).get() / rowsum_vec_d.shape[0]


    def HDD_d(self, t_0, t_step, T_0, sp ):

        ttdp = (self.date_p - self.date_o).days / 365.25
        t_v = t_0 + t_step * np.arange(self.N_step) 
        T_m_v = self.T_m (ttdp + t_v )
        T_m_d1 = self.T_m_der (ttdp + t_v) # vector as well

        T_sm_d = self.T_sim_inn_d (T_m_v, T_m_d1, t_step, sp ) # T simulated matrix 
        return self.HDD_payoff_d( T_sm_d )

    def HDDO_payoff(self, K, T_sm, cp_ind = "c"):
        return np.average (np.maximum (np.sum (np.maximum (65. - T_sm , 0.), axis = 1) - K, 0.) )



    # HDD option real pricing 
    def HDDO_real (self, K, nb_months, sp_l, HDD_date_l, cp_ind = "c"):
    
        mi = self.months_index (HDD_date_l, sp_l[0]) # just need month_decom 
        mi_dec = self.month_decomp (mi["month_decomp"])
        sim_size = self.Z_m.shape[1]
        hdd_sim = np.zeros (( nb_months, sim_size))

        for n in np.arange (nb_months):
            t_month_start = self.add_months (self.date_s, n) 
            t_month_start_num = (t_month_start - self.date_p).days / 365.
            # T_0 = T_m_real (t_month_start, date_o, hp)
            t_step = 1. / 365.
            # N_step = (add_months(t_month_start, 1) - t_month_start).days 
            # COMPLETELY WRONG WRONG WRONG, PREVIOUS LINE IS CORRECT 
            # N_step = 31 
            sp = sp_l[self.month_into_sigma(n,mi_dec)] # mapping into sp_l[ n
            ttdp = (self.date_p - self.date_o).days / 365.25
            
            # following 3 lines used in both cpu/gpu comp.
            t_v = t_month_start_num + t_step * np.arange(self.N_step) 
            T_m_v = self.T_m (ttdp + t_v )
            T_m_d1 = self.T_m_der (ttdp + t_v ) # vector as well

            if self.gpu_ind == False:
                # Z_m = np.random.normal(size=(sim_size,N_step))
                T_sim = self.T_sim_inn (T_m_v, T_m_d1, t_step, sp ) 
                hdd_sim[n,:] = np.sum (np.maximum (65. - T_sim , 0.), axis = 0)

            else:
                T_sim_d = self.T_sim_inn_d ( T_m_v, T_m_d1, t_step, sp )
                T_sim_d = 65. - T_sim_d
                maximum_cuda(T_sim_d)
                rowsum_vec_d = colsum_cuda_last (T_sim_d)
                # write_vec_in_mat_col ( rowsum_vec_d, hdd_sim_d, n) # impl. : hdd_sim_d[n,:] = rowsum_vec_d
                hdd_sim[n,:] = rowsum_vec_d.get()

        return np.average (np.maximum ( np.sum (hdd_sim, axis =0 ) - K, 0.))


    def HDDO_histo (self, K, nb_months, HDD_date_l, cp_ind = 'c'):
        return self.HDDO_real (K, nb_months, [(0.,0.)] * len (HDD_date_l), HDD_date_l, 
                               cp_ind)




    def HDDO_payoff_d (self, K, T_sm_d, cp_ind = "c"):
        """
        same as above, just that most operations are performed on the device
        """
        T_sm_tmp_d = 65. - T_sm_d
        maximum_cuda(T_sm_tmp_d)
        colsum_cuda(T_sm_tmp_d)
        HDD_payoff_tmp = T_sm_tmp_d - K

        # implements: return np.average (np.maximum (HDD_payoff_tmp.get(), 0.) )
        return average_reduction (maximum_cuda (HDD_payoff_tmp)).get / HDD_payoff_tmp.shape[0]

    def T_par_inn (self, T_m_v, T_m_d, dt, sp):
        """
        fast simulation of the weather model partial innovations  
        """
        a = self.hp[5]
        sigma, sigma_lam = sp
        
        res = (T_m_d + a * T_m_v - sigma_lam ).reshape((len(T_m_d),1)) * dt + \
            sigma * np.sqrt (dt) * self.Z_m

        return res


    def T_par_inn_d (self, T_m_v, T_m_d1, dt, sp):
        """
        same as T_par_inn, except that it runs on GPU device 
        """

        a = self.hp[5]
        sigma, sigma_lam = sp

        # this implements ( *a _has_ be on the right )
        v = (T_m_d1 + T_m_v * a - sigma_lam ) * dt
    
        # Z_m_d NEEDS to have 0's in the first column
        # v = np.append ( T_0, ( T_m_d + a * T_m_v  - sigma_lam ) * dt )

        inn_d = self.Z_m_d * (sigma * np.sqrt (dt)) # mult. _has_ to be on the right (problems with pycuda)
        vtpm_cols ( v, inn_d, 'p' ) # inn_d <- v_d + inn_d 

        return inn_d



    
    def T_sim_inn (self, T_m_v, T_m_d, t_step, sp):
        """
        simulation from partial innovations, FAST simulation 
        """
        # t_v = t_0 + t_step * np.arange(N_step) 
        T_s_1 = self.T_par_inn (T_m_v, T_m_d, t_step, sp)
    
        # THERE SHOULD BE T_0, perhaps we can circumvent this 
        # T_s_1 = np.append (T_0 * np.ones((Z_m.shape[0],1)), T_s_1, axis=1)
        a = self.hp[5]
        T_s_2 = T_s_1 * (1-a * t_step)**np.arange(self.N_step-1,-1,-1).reshape((self.N_step,1))

        return np.cumsum(T_s_2,axis=0) / (1-a*t_step)**np.arange(self.N_step-1,-1,-1).reshape((self.N_step,1))

    def T_sim_inn_d (self, T_m_v_v, T_m_d_v, t_step, sp ):
        # this below is inlined directly below 
        # T_s_1_d = T_par_inn_d ( T_m_v_d, T_m_d_d, t_step, sp, hp, Z_m_d, date_p, date_o)
    
        a = self.hp[5]
        sigma, sigma_lam = sp
        # this implements ( *a _has_ be on the right )
        v = (T_m_d_v + T_m_v_v * a - sigma_lam ) * t_step
        T_s_1_d = self.Z_m_d * (sigma * np.sqrt (t_step)) # mult. _has_ to be on the right (problems with pycuda)
        vtpm_cols ( v, T_s_1_d, 'p' ) # inn_d <- v_d + inn_d 

        # appending from T_sim_inn is done in T_par_inn_d directly 
        # implements: T_s_1_d =  (1-a * t_step)**arange(N_step,-1,-1) * T_s_1_d 
        # TO IMPROVE BELOW:
        # N_step = 31 in this case, otherwise can be different
        # mult2 = gpa.to_gpu( (1-a * t_step)**(- arange(N_step, -1,-1)).astype(np.float32) )

        range_1 = ( 1. - a * t_step )**np.arange(self.N_step - 1, -1,-1)
        vtpm_cols ( range_1, T_s_1_d, 't' ) # multiply
        
        # following 2 statements implement the following 
        # np.cumsum(T_s_1_d.get(),axis=1)/ (1-a*t_step)**arange(N_step,-1,-1)
        colsum_cuda (T_s_1_d) # cumsum on cuda
        vtpm_cols ( 1./range_1, T_s_1_d, 't')
        
        return T_s_1_d

