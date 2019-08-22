import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import theano
import csv
import os
import timeit
from datetime import date

import warnings
warnings.filterwarnings('ignore')

def get_data(name):
    yreader = csv.reader(open(name + ".csv"))
    Y = np.array([row for row in yreader]).astype(float)
    return Y

def get_func(name, n):
    sreader = csv.reader(open(name + ".csv"))
    mFunc = np.array([row for row in sreader]).astype(float)
    func_new = np.array(mFunc[0, 0:n*(n-1)//2])
    func_temp = np.triu(np.ones([n, n]),1)
    func_temp[func_temp==1] = func_new
    Func_mat = func_temp.T + np.eye(n) + func_temp
    return Func_mat

def get_struct(name, n):
    sreader = csv.reader(open(name + ".csv"))
    S_read = np.array([row for row in sreader]).astype(float)
    struct_new = np.array(S_read[0:n*(n-1)//2, 0])
    Struct_temp = np.triu(np.ones([n, n]), 1)
    Struct_temp[Struct_temp ==1] = struct_new
    Struct_mat = Struct_temp.T + np.eye(n) + Struct_temp
    return Struct_mat

def get_dist(name, n):
    Dist = []
    for i in range(1, n+1):
        distreader = csv.reader(open(name + "_" + str(i) + ".csv"))
        Dist.append(np.array([row for row in distreader]).astype(float))
    return Dist

def run_model(index, slide_index, Y, mFunc, Struct, Dist, n, kernel, lambdaw, Kf, sample_size, tune_size):
    """
    index: index of object data
    slide_index: index of slide window
    Y: time-series data
    mFunc: functional connectivity
    Struct: structural connectivity
    Dist: distribution matrix of n ROIs 
    n: ROI number
    kernel: "exponential" or "gaussian" or "matern52" or "matern32"
    lambdaw: weighted parameter
    kf: weighted parameter
    sample_size: NUTS number
    tune_size: burning number
    """
    m = Dist[0].shape[0]
    k = Y.shape[1]
    n_vec= n*(n+1)//2
    
    Y_mean = []
    for i in range(n):
        Y_mean.append(np.mean(Y[i*m:(i+1)*m, 0]))
    Y_mean = np.array(Y_mean)
    
    with pm.Model() as model_generator:
        
        # convariance matrix
        log_Sig = pm.Uniform("log_Sig", -8, 8, shape = (n, ))
        SQ = tt.diag(tt.sqrt(tt.exp(log_Sig)))
        Func_Covm = tt.dot(tt.dot(SQ, mFunc), SQ)
        Struct_Convm = tt.dot(tt.dot(SQ, Struct), SQ)
        
        # double fusion of structural and FC
        L_fc_vec = tt.reshape(tt.slinalg.cholesky(tt.squeeze(Func_Covm)).T[np.triu_indices(n)], (n_vec, ))
        L_st_vec = tt.reshape(tt.slinalg.cholesky(tt.squeeze(Struct_Convm)).T[np.triu_indices(n)], (n_vec, ))
        Struct_vec = tt.reshape(Struct[np.triu_indices(n)], (n_vec, ))
        rhonn = Kf*( (1-lambdaw)*L_fc_vec + lambdaw*L_st_vec ) + \
            (1-Kf)*( (1-Struct_vec*lambdaw)*L_fc_vec + Struct_vec*lambdaw*L_st_vec )
        
        # correlation
        Cov_temp = tt.triu(tt.ones((n,n)))
        Cov_temp = tt.set_subtensor(Cov_temp[np.triu_indices(n)], rhonn)
        Cov_mat_v = tt.dot(Cov_temp.T, Cov_temp)
        d = tt.sqrt(tt.diagonal(Cov_mat_v))
        rho = (Cov_mat_v.T/d).T/d
        rhoNew = pm.Deterministic("rhoNew", rho[np.triu_indices(n,1)])
       
        # temporal correlation AR(1)
        phi_T = pm.Uniform("phi_T", 0, 1, shape = (n, ))
        sigW_T = pm.Uniform("sigW_T", 0, 100, shape = (n, ))
        B = pm.Normal("B", 0, 100, shape = (n, ))
        muW1 = Y_mean - B # get the shifted mean
        mean_overall = muW1/(1.0-phi_T) # AR(1) mean
        tau_overall = (1.0-tt.sqr(phi_T))/tt.sqr(sigW_T) # AR (1) variance
        W_T = pm.MvNormal("W_T", mu = mean_overall, tau = tt.diag(tau_overall), shape = (k, n))
        
        # add all parts together
        one_m_vec = tt.ones((m, 1))
        one_k_vec = tt.ones((1, k))
        D = pm.MvNormal("D", mu = tt.zeros(n), cov = Cov_mat_v, shape = (n, ))
        phi_s = pm.Uniform("phi_s", 0, 20, shape = (n, ))
        spat_prec = pm.Uniform("spat_prec", 0, 100, shape = (n, ))
        H_base = pm.Normal("H_base", 0, 1, shape = (m, n))
        
        Mu_all = tt.zeros((m*n, k))
        if kernel == "exponential":
            for i in range(n):
                r = Dist[i]*phi_s[i]
                H_temp = tt.sqr(spat_prec[i])*tt.exp(-r)
                L_H_temp = tt.slinalg.cholesky(H_temp)
                Mu_all_update = tt.set_subtensor(Mu_all[m*i:m*(i+1), :], B[i] + D[i] + one_m_vec*W_T[:,i] + \
                    tt.dot(L_H_temp, tt.reshape(H_base[:,i], (m, 1)))*one_k_vec)
                Mu_all = Mu_all_update
        elif kernel == "gaussian":
            for i in range(n):
                r = Dist[i]*phi_s[i]
                H_temp = tt.sqr(spat_prec[i])*tt.exp(-tt.sqr(r)*0.5)
                L_H_temp = tt.slinalg.cholesky(H_temp)
                Mu_all_update = tt.set_subtensor(Mu_all[m*i:m*(i+1), :], B[i] + D[i] + one_m_vec*W_T[:,i] + \
                    tt.dot(L_H_temp, tt.reshape(H_base[:,i], (m, 1)))*one_k_vec)
                Mu_all = Mu_all_update
        elif kernel == "matern52":
            for i in range(n):
                r = Dist[i]*phi_s[i]
                H_temp = tt.sqr(spat_prec[i])*((1.0+tt.sqrt(5.0)*r+5.0/3.0*tt.sqr(r))*tt.exp(-1.0*tt.sqrt(5.0)*r))
                L_H_temp = tt.slinalg.cholesky(H_temp)
                Mu_all_update = tt.set_subtensor(Mu_all[m*i:m*(i+1), :], B[i] + D[i] + one_m_vec*W_T[:,i] + \
                    tt.dot(L_H_temp, tt.reshape(H_base[:,i], (m, 1)))*one_k_vec)
                Mu_all = Mu_all_update
        elif kernel == "matern32":
            for i in range(n):
                r = Dist[i]*phi_s[i]
                H_temp = tt.sqr(spat_prec[i])*(1.0+tt.sqrt(3.0)*r)*tt.exp(-tt.sqrt(3.0)*r)
                L_H_temp = tt.slinalg.cholesky(H_temp)
                Mu_all_update = tt.set_subtensor(Mu_all[m*i:m*(i+1), :], B[i] + D[i] + one_m_vec*W_T[:,i] + \
                    tt.dot(L_H_temp, tt.reshape(H_base[:,i], (m, 1)))*one_k_vec)
                Mu_all = Mu_all_update
        
        sigma_error_prec = pm.Uniform("sigma_error_prec", 0, 100)
        Y1 = pm.Normal("Y1", mu = Mu_all, sd = sigma_error_prec, observed = Y)
    
    with model_generator:
        step = pm.NUTS()
        trace = pm.sample(sample_size, step = step, tune = tune_size, chains = 1)

    # save as pandas format and output the csv file
    save_trace = pm.trace_to_dataframe(trace)
    save_trace.to_csv(out_dir + date.today().strftime("%m_%d_%y") + \
        "_sample_size_" + str(sample_size) + "_index_" + str(index) + "_slide_index_" + str(slide_index) +".csv")

# initializing parameters
index_list = [8007, 8012, 8049, 8050, 8068, 8072, 8077, 8080, \
              8098, 8107, 8110, 8146, 8216, 8244, 8245, 8246, \
              8248, 8250, 8253, 8256, 8257, 8261, 8262, 8263, \
              8264, 8265, 8266, 8273, 8276, 8279, 8280, 8282, \
              8283, 8284, 8285, 8288, 8290, 8292, 8293, 8295, \
              8299]

in_dir = "/Users/ruiwang/source/doublefusion/simulation/data/" # in_dir: set up work directory
out_dir = "/Users/ruiwang/source/doublefusion/simulation/results/" # out_dir: save the trace as csv in the out directory
data_filename = "ROI_timeseries_data" # data_filename: filename for time series data
func_filename = "DMN_MeanFunctional_Connectivity" # func_filename: filename for functional connectivity
struct_filename = "DMN_StructuralConnectivity" # struct_filename: filename for structural connectivity
dist_filename = "distMatrix_ROI" # dist_filename: filename for distribution matrix of n ROIs 

kernel = "exponential"
n = 14
sample_size = 1000
tune_size = 500 ## make it smaller for dynamic FC

# run the model
for index in index_list:
    # obtain the previous results
    df_previous = pd.read_csv(out_dir + date.today().strftime("%m_%d_%y") + "_sample_size_" + str(sample_size) + "_index_" + str(index) + ".csv")
    # fix lambdaw and Kf
    # lambdaw obtained from previsous results
    lambdaw = np.squeeze(df_previous[df_previous.columns[df_previous.columns.str.startswith("lambdaw")]].median(axis = 1).values())
    # Kf obtained from previous results
    Kf = np.squeeze(df_previous[df_previous.columns[df_previous.columns.str.startswith("Kf")]].median(axis = 1).values())
    
    os.chdir(in_dir + str(index))
    Y = get_data(data_filename)
    mFunc = get_func(func_filename, n)
    Struct = get_struct(struct_filename, n)
    Dist = get_dist(dist_filename, n)
    t_interval = 25 # set the interval length to be 25
    t_total = Y.shape[1] # the total is 150
    for slide_index in range(t_total-t_interval):
        Y_slide = Y[:, slide_index:(slide_index+t_interval)]
        run_model(index, slide_index, Y_slide, mFunc, Struct, Dist, n, kernel, lambdaw, Kf, sample_size, tune_size)