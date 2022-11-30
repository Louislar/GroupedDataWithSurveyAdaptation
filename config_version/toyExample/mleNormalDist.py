'''
計算toy example的MLE估計
'''
import numpy as np 
import scipy.stats as st
from scipy.optimize import minimize
import pandas as pd

IntervalSet1 = [
    [0, 6], 
    [6, 10]
]
IntervalSet2 = [
    [0, 3],
    [3, 4],
    [4, 6],
    [6, 8], 
    [8, 10]
]

MidpointsSet1 = [3, 8]
MidpointsSet2 = [1.5, 3.5, 5, 7, 9]

relFreqQ1 = [0.8413, 0.1587]
relFreqQ2 = [0.0227, 0.1359, 0.6828, 0.1573, 0.0013]
# 中間少0.01, 最左與最右加0.05
relFreqVAMAdj = [0.0277, 0.1359, 0.6728, 0.1573, 0.0063]

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
        ## TODO: 但是回傳-inf不是個好事, 該如何解決?
        logProb +=  _freq_i * (np.log(_cProb) if _cProb!=0 else -1e5)
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
            lb=0, 
            ub=10
        ),
        x0=np.array([3.7935, 1]),
        method='SLSQP',
        bounds=[(1e-6, 1e6), (-1e6, 1e6)]
    )
    mleQ1 = [mleQ1.x[0], mleQ1.x[1]]
    print('Q1 mean: ', computeTrunNormMean(mleQ1[0], mleQ1[1], 0, 10))
    print('Q1 param: ', mleQ1)
    ## mle of q2 
    mleQ2 = minimize(
        lambda x: truncatedNormalIntervalLogCDF(
            dataIn=freqQ2,
            qInterval=IntervalSet2,
            mean=x[0],
            std=x[1], 
            lb=0, 
            ub=10
        ),
        x0=np.array([5.035, 1]),
        method='SLSQP',
        bounds=[(1e-6, 1e6), (-1e6, 1e6)]
    )
    mleQ2 = [mleQ2.x[0], mleQ2.x[1]]
    print('Q2 mean: ', computeTrunNormMean(mleQ2[0], mleQ2[1], 0, 10))
    print('Q2 param: ', mleQ2)
    ## mle of q1 adjusted by VAM 
    mleVAMAdj = minimize(
        lambda x: truncatedNormalIntervalLogCDF(
            dataIn=freqVAMAdj,
            qInterval=IntervalSet2,
            mean=x[0],
            std=x[1], 
            lb=0, 
            ub=10
        ),
        x0=np.array([5.038, 1]),
        method='SLSQP',
        bounds=[(1e-6, 1e6), (-1e6, 1e6)]
    )
    mleVAMAdj = [mleVAMAdj.x[0], mleVAMAdj.x[1]]
    print('Q1 adjusted by VAM mean: ', computeTrunNormMean(mleVAMAdj[0], mleVAMAdj[1], 0, 10))
    print('Q1 adjusted by VAM param: ', mleVAMAdj)

    ## Midpoint result
    ## Q1 
    midQ1 = [i*j for i, j in zip(MidpointsSet1, relFreqQ1)]
    ## Q2 
    midQ2 = [i*j for i, j in zip(MidpointsSet2, relFreqQ2)]
    ## Q1 adjust br VAM 
    midVAMAdj = [i*j for i, j in zip(MidpointsSet2, relFreqVAMAdj)]
    print('midpoint Q1: ', sum(midQ1))
    print('midpoint Q2: ', sum(midQ2))
    print('midpoint Q1 adjusted by VAM: ', sum(midVAMAdj))

# 產生frequency vector based on 給定的參數與underlying distribution
def generateRelativeFreq():
    my_a = 0
    my_b = 10
    def cdfOfTruncNorm(x):
        return st.truncnorm.cdf(x=x, a=(my_a-5)/1, b=(my_b-5)/1, loc=5, scale=1)
    valList = [cdfOfTruncNorm(i+1) for i in range(-1, 10+1)]
    print(valList[6])

    print('======= Q_1 =======')
    print(valList[6]-valList[0])
    print(valList[10]-valList[6])

    print('======= Q_2 =======')
    print(valList[3]-valList[0])
    print(valList[4]-valList[3])
    print(valList[6]-valList[4])
    print(valList[8]-valList[6])
    print(valList[10]-valList[8])
    pass

if __name__=='__main__':
    main()
    pass