'''
計算toy example的MLE估計
'''
import numpy as np 
import scipy.stats as st
from scipy.optimize import minimize
import pandas as pd

IntervalSet1 = [
    [-1e6, 0], 
    [0, 1e6]
]
IntervalSet2 = [
    [-1e6, -1],
    [-1, 1],
    [1, 1e6]
]

MidpointsSet1 = [-5e5, 5e5]
MidpointsSet2 = [-75e4, 0, 75e4]

relFreqQ1 = [0.5005, 0.4995]
relFreqQ2 = [0.3335, 0.3335, 0.3330]
relFreqVAMAdj = [0.3335, 0.3334, 0.3331]

sampleSize = 1e4

def truncatedNormalIntervalLogCDF(dataIn, qInterval, mean, std, lb, ub):
    '''
    Objective: 
        計算truncated normal distribution log CDF形式的數值, 
        但是機率累積的區間是給定的 (傳統CDF是從0開始到給定數值)
    Input: 
    :dataIn: 各個區間的frequency. 固定數字編號從1開始.
    :qInterval: 
    :mean: truncated normal distribution parameter mu
    :std: truncated normal distribution parameter sigma  
    :lb: truncated normal lower bound (target distribution lower bound)
    :ub: truncated normal upper bound

    Output:
    :logProb: 給定parameter下出現dataIn的機率的負數 (*-1)
    '''
    logProb = 0
    for i, _freq_i in enumerate(dataIn):
        _lb = qInterval[i][0]
        _ub = qInterval[i][1]
        _cProb = st.truncnorm.cdf(_ub, a=(lb-mean)/std, b=(ub-mean)/std, loc=mean, scale=std) - \
            st.truncnorm.cdf(_lb, a=(lb-mean)/std, b=(ub-mean)/std, loc=mean, scale=std)
        ## 這邊若是機率為0導致log(機率)為-inf, 
        ## 代表給定的參數產生的distribution在此區間產生樣本的機率為0
        ## 代表這並非我們想要的distribution
        ## TODO: 但是回傳inf不是個好事, 該如何解決?
        logProb +=  _freq_i * (np.log(_cProb) if _cProb!=0 else -1e10)
    # print(logProb)
    return logProb * (-1)

def computeTrunNormMean(mean, std, lb, ub):
    '''
    計算truncated normal的mean
    Input:
    :mean: mean of underlying normal distribution
    :std: std of underlying normal distribution
    :lb: lower bound 
    :ub: upper bound 
    '''
    alpha = (lb - mean) / std
    beta = (ub - mean) / std
    truncMean = mean + \
        std * \
        ((st.norm.pdf(x=alpha, loc=0, scale=1) - st.norm.pdf(x=beta, loc=0, scale=1)) / (st.norm.cdf(x=beta, loc=0, scale=1) - st.norm.cdf(x=alpha, loc=0, scale=1)))
    return truncMean

def main():
    freqQ1 = [round(i*sampleSize) for i in relFreqQ1]
    freqQ2 = [round(i*sampleSize) for i in relFreqQ2]
    freqVAMAdj = [round(i*sampleSize) for i in relFreqVAMAdj]

    ## mle of q1 
    mleQ1 = minimize(
        lambda x: truncatedNormalIntervalLogCDF(
            dataIn=freqQ1,
            qInterval=IntervalSet1,
            mean=x[0],
            std=x[1], 
            lb=-1e6, 
            ub=1e6
        ),
        x0=np.array([1, 1]),
        method='SLSQP',
        bounds=[(-1e6, 1e6), (-1e6, 1e6)]
    )
    mleQ1 = [mleQ1.x[0], mleQ1.x[1]]
    print(computeTrunNormMean(mleQ1[0], mleQ1[1], -1e6, 1e6))
    print(mleQ1)
    ## mle of q2 
    mleQ2 = minimize(
        lambda x: truncatedNormalIntervalLogCDF(
            dataIn=freqQ2,
            qInterval=IntervalSet2,
            mean=x[0],
            std=x[1], 
            lb=-1e6, 
            ub=1e6
        ),
        x0=np.array([1, 1]),
        method='SLSQP',
        bounds=[(-1e6, 1e6), (-1e6, 1e6)]
    )
    mleQ2 = [mleQ2.x[0], mleQ2.x[1]]
    print(computeTrunNormMean(mleQ2[0], mleQ2[1], -1e6, 1e6))
    print(mleQ2)
    ## mle of q1 adjusted by VAM 

    
    pass

if __name__=='__main__':
    main()
    pass