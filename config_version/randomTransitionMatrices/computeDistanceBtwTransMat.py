'''
計算random generate的transition matrices之間的距離, Linfinity
'''
import numpy as np 
import pandas as pd 
from itertools import combinations

rndTransMatCount = 3

def main(rndTransMatCount):
    '''
    1. read all the transition matrices
    2. compute L infinity between trans mats
    3. print the result in the console
    '''
    # 1. 
    rndTransMats = []
    for i in range(rndTransMatCount):
        rndTransMats.append(
            pd.read_csv(
                'random_transition_matrix_{0}.csv'.format(i)
            ).values
        )
    # 2. 
    indCombination = combinations(range(rndTransMatCount), 2)
    distances = []
    for i, j in indCombination:
        _tmp = np.abs(rndTransMats[i] - rndTransMats[j])
        _tmp = np.max(_tmp)
        distances.append(_tmp)
        print(i, ', ', j)
        print(_tmp)
    print('average distance: ', np.mean(distances))
    pass

if __name__=='__main__':
    main(rndTransMatCount)