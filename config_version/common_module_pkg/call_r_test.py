import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# ro.globalenv['var01'] = 1.5
# ro.globalenv['var02'] = ro.IntVector([1, 2, 3, 4, 5])



# print(type(ro.r('var02')))
# print(ro.globalenv['var02'])
# print(ro.globalenv['var02'][0])
# print(type(ro.globalenv['var02'][0]))

def estimate_gamma_distribution_by_r(datta): 
    '''
    根據傳入的資料，使用MLE計算最接近的gamma distribution的參數
    Output: 
    :alpha: 估計出的gamma的參數之一
    :theta: 估計出的gamma的參數之一
    '''
    print('------estimate gamma (R)------')
    ro.globalenv['D'] = ro.FloatVector(datta)
    ro.r('momalpha <- mean(D)^2/var(D) \n\
        mombeta <- var(D)/mean(D)')
    ro.r('gmll <- function(theta,datta) \n\
    { \n\
    a <- theta[1]; b <- theta[2] \n\
    n <- length(datta); sumd <- sum(datta) \n\
    sumlogd <- sum(log(datta)) \n\
    gmll <- n*a*log(b) + n*lgamma(a) + sumd/b - (a-1)*sumlogd \n\
    gmll \n\
    }')
    # ro.r('gammasearch = nlm(gmll,c(momalpha,mombeta),hessian=T,datta=D)')
    # ro.r('install.packages("nloptr")')
    ro.r('library(nloptr)')
    ro.r('gammasearch = nloptr::bobyqa(fn=gmll,x0=c(momalpha,mombeta), datta=D, lower=c(0.0001, 0.0001), upper=c(Inf, Inf))')
    # ro.r('gammasearch_result <- gammasearch$estimate')
    ro.r('gammasearch_result <- gammasearch$par')
    gamma_search_result = ro.globalenv['gammasearch_result']
    # print(ro.globalenv['gammasearch'])

    print('nlm initial alpha: ', ro.globalenv['momalpha'])
    print('nlm initial beta: ', ro.globalenv['mombeta'])

    return gamma_search_result[0], gamma_search_result[1]

if __name__ == "__main__":
    alpha_hat, beta_hat = estimate_gamma_distribution_by_r([1, 2, 3])
    print('alpha_hat: ', alpha_hat)
    print('beta_hat: ', beta_hat)