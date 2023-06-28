'''
尋找美兆資料cohort的transition matrix的規律, 
利用該規律randomly generate新的transition matrices
注意, 固定random seed, 方便重現結果
'''
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

np.random.seed(0)

mjTransMatPath = '../mj_gamma_study/multi_year_patient_data/'
transMatYears = [i for i in range(1998, 2007+1)]
transMatFilePrefix = 'real_transition_matrix_{0}_{1}.csv'
outputPath = './'
outputFilePrefix = 'random_transition_matrix_{0}.csv'

def generateRandomTransMat():
    '''
    Goal: find the pattern of the differences and 
        use the pattern to generate new transition matrix
    1. read all the MJ transition matrices
    2. compute difference between identity matrix and MJtransition matrices
    3. plot difference to find pattern
    4. compute diagonal difference range 
    5. distribution of non-diagonal values in each column
    -----
    6. generate random difference of diagonal values 
    7. generate random weights for non-diagonal distributions and do weighted average 
        then normalize the result (make it sum to 1)
    8. multiply the difference of diagonal value and the non-diagonal distribution
        and get the difference of non-diagnoal values
    9. combine diagonal and non-diagonal values 
    10. check if the result matrix is valid. (加總column, 要是0)
    '''
    # 1. 
    transMats = []
    for _yr in transMatYears:
        transMats.append(pd.read_csv(
            os.path.join(mjTransMatPath, transMatFilePrefix.format(_yr, _yr+1)),
            index_col=0
        ).T)
    # print(transMats[0])

    # 2. 
    transMatsArr = [i.values for i in transMats]
    identityMat = np.identity(transMatsArr[0].shape[0])
    transMatsDiff = [i - identityMat for i in transMatsArr]
    print(transMatsArr[0])
    print(transMatsArr[1])
    print(transMatsArr[2])
    # print(identityMat)
    # print(transMatsDiff[0])
    # print(transMatsDiff[1])
    # print(transMatsDiff[2])

    # 3.
    plt.figure()
    # plt.plot([i[0,0] for i in transMatsDiff], [0 for i in transMatsDiff], '.')
    ## 觀察與時間有沒有相關 (單看最左上角的value, 不與時間相關)
    plt.plot([i for i in range(len(transMatsDiff))], [i[0,0] for i in transMatsDiff], '.')
    # plt.show()

    # 4. range of difference of diagonal values 
    diagonalRange = []
    for i in range(transMatsDiff[0].shape[0]):
        _val = []
        for j in range(len(transMatsDiff)):
            _val.append(transMatsDiff[j][i, i])
        diagonalRange.append(
            [np.min(_val), np.max(_val)]
        )
    # print(diagonalRange[0])

    # 5. distribution of non-diagonal values 
    nonDiagonalVal = []
    identityBoolMat = identityMat==0
    # 去除diagonal的數值, 並且保留原始column數值位置
    for i in range(len(transMatsDiff)):
        nonDiagonalVal.append(
            transMatsDiff[i].T[identityBoolMat].reshape(-1, transMatsDiff[i].shape[0]-1).T
        )
    # 轉換每一個數值數值為, 他佔單一column總數值的比例
    nonDiagonalDistribution = [i/i.sum(axis=0) for i in nonDiagonalVal]
    
    # print(nonDiagonalVal[0])
    # print(nonDiagonalVal[1])
    # print(nonDiagonalVal[2])

    # print(nonDiagonalDistribution[0])
    # print(nonDiagonalDistribution[1])
    # print(nonDiagonalDistribution[2])

    # 6. generate random difference of diagonal values 
    randomDiagonalValue = []
    for _lowerUpperBounds in diagonalRange:
        randomDiagonalValue.append(
            np.random.uniform(_lowerUpperBounds[0], _lowerUpperBounds[1])
        )
    # print(randomDiagonalValue[0])

    # 7. generate random weights for non-diagonal distributions and do weighted average 
    #       then normalize the result (make it sum to 1)
    ## generate random range 
    weights = np.random.uniform(0, 1, size=len(nonDiagonalDistribution))
    weights = weights/sum(weights)
    ## weighted average the non-diagonal distribution
    weightedAvgDist = np.zeros_like(nonDiagonalDistribution[0])
    for i, _dist in enumerate(nonDiagonalDistribution):
        weightedAvgDist = weightedAvgDist + _dist * weights[i]
    ## normalize the result (make each column sum to 1)
    weightedAvgDist = weightedAvgDist/weightedAvgDist.sum(axis=0)
    # print(weights)
    # print(weightedAvgDist)
    # 8. compute non-diagonal values 
    randomNonDiagonalVal = weightedAvgDist * randomDiagonalValue * -1
    # print(randomNonDiagonalVal)

    # 9. combine non-diagonal and diagonal values 
    randomTransMat = np.zeros_like(transMatsArr[0])
    randomTransMat[identityMat==1]=randomDiagonalValue
    randomTransMat.T[identityMat==0] = randomNonDiagonalVal.flatten('F')
    # print(randomTransMat)

    # 10. 每一個column加總, 確認有無出錯. 正常加總數值要是1
    print(randomTransMat.sum(axis=0))

    # 11. 與identity matrix相加, 得到最終的transition matrix
    randomTransMat = randomTransMat + identityMat
    print(randomTransMat)

    return randomTransMat

if __name__=='__main__':
    ## TODO: randomly generate 3 transition matrices and store into .csv
    numOfRandomTransMat = 3
    for i in range(numOfRandomTransMat):
        _transMat = generateRandomTransMat()
        _transMatDf = pd.DataFrame(_transMat).to_csv(
            os.path.join(outputPath, outputFilePrefix.format(i)), 
            index=False
        )
    pass