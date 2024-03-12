import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

# cochrane-orcutt / prais-winsten with given AR(1) rho, 
# derived from ols model, default to cochrane-orcutt 
def ols_ar1(model,rho,drop1=False):
    x = model.exog
    y = model.endog
    ystar = y[1:]-rho*y[:-1]
    xstar = x[1:,]-rho*x[:-1,]
    if drop1 == False:
        ystar = np.append(np.sqrt(1-rho**2)*y[0],ystar)
        xstar = np.append([np.sqrt(1-rho**2)*x[0,]],xstar,axis=0)
    model_ar1 = sm.OLS(ystar,xstar).fit()
    return(model_ar1)

# cochrane-orcutt / prais-winsten iterative procedure
# default to cochrane-orcutt (drop1=True)
def OLSAR1(model,drop1=False):
    x = model.exog
    y = model.endog
    res = model.fit()
    e = res.resid
    e1 = e[:-1]; e0 = e[1:]
    rho0 = np.dot(e1,e0)/np.dot(e1,e1)
    rdiff = 1.0
    while(rdiff>1.0e-5):
        model1 = ols_ar1(model,rho0,drop1)
        e = model1.resid
        e1 = e[:-1]; e0 = e[1:]
        rho1 = np.dot(e1,e0)/np.dot(e1,e1)
        rdiff = np.sqrt((rho1-rho0)**2)
        rho0 = rho1
        print('Rho = ', rho0)
    # pint final iteration
    # print(sm.OLS(e0,e1).fit().summary())
    model1 = ols_ar1(model,rho0,drop1)
    return(model1)

def ols_ar1_panel_flat(model,group_mask,rho,drop1=False):
    group_mask = group_mask.reset_index(drop=True)
    grouped = group_mask.groupby(by=group_mask)
    xstar = []
    ystar = []
    for group_indices in grouped.indices.values():
        x = model.exog[group_indices]
        y = model.endog[group_indices]
        if drop1 == False:
            ystar.append([np.sqrt(1-rho**2)*y[0]])
            xstar.append([np.sqrt(1-rho**2)*x[0,]])
        ystar.append(y[1:]-rho*y[:-1])
        xstar.append(x[1:,]-rho*x[:-1,])
    xstar = np.vstack(xstar)
    ystar = np.concatenate(ystar)
    model_ar1 = sm.OLS(ystar,xstar).fit()
    return(model_ar1)

# for panel data
# dummy_matrix is a mask upon objects
# if model.exog is N x K, then dummy_matrix is K x N
def OLSAR1_panel_flat(model,group_mask: pd.Series,drop1=False):
    group_mask = group_mask.reset_index(drop=True)
    grouped = group_mask.groupby(by=group_mask)
    def calc_rho(e):
        rho_num = 0
        rho_denom = 0
        for group_indices in grouped.indices.values():
            e_group = e[group_indices]
            e1 = e_group[:-1]; e0 = e_group[1:]
            rho_num += np.dot(e1,e0)
            rho_denom += np.dot(e0,e0)
        rho = rho_num / rho_denom
        return rho

    x = model.exog
    y = model.endog
    result = model.fit()
    rho0 = calc_rho(result.resid)
    rdiff = 1.0
    while(rdiff>1.0e-5):
        model1_res = ols_ar1_panel_flat(model,group_mask,rho0,drop1)
        rho1 = calc_rho(model1_res.resid)
        rdiff = np.sqrt((rho1-rho0)**2)
        rho0 = rho1
        print('Rho = ', rho0)
    # pint final iteration
    # print(sm.OLS(e0,e1).fit().summary())
    model1 = ols_ar1(model,rho0,drop1)
    return(model1)

import numpy as np


def create_data(examples=50, features=5, upper_bound=10, outliers_fraction=0.1, extreme=False):
    '''
    This method for testing (i.e. to generate a 2D array of data)
    '''
    data = []
    magnitude = 4 if extreme else 3
    for i in range(examples):
        if (examples - i) <= round((float(examples) * outliers_fraction)):
            data.append(np.random.poisson(upper_bound ** magnitude, features).tolist())
        else:
            data.append(np.random.poisson(upper_bound, features).tolist())
    return np.array(data)


def MahalanobisDist(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            vars_mean = []
            for i in range(data.shape[0]):
                vars_mean.append(list(data.mean(axis=0)))
            diff = data - vars_mean
            md = []
            for i in range(len(diff)):
                md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))

            if verbose:
                print("Covariance Matrix:\n {}\n".format(covariance_matrix))
                print("Inverse of Covariance Matrix:\n {}\n".format(inv_covariance_matrix))
                print("Variables Mean Vector:\n {}\n".format(vars_mean))
                print("Variables - Variables Mean Vector:\n {}\n".format(diff))
                print("Mahalanobis Distance:\n {}\n".format(md))
            return md
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")



def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def isMahalanobisOutlier(data, threshold=0.1):
    square_of_mahalanobis_distances = np.array(MahalanobisDist(data)) ** 2
    degree_of_freedom = data.shape[1]

    pvalues = 1 - chi2.cdf(square_of_mahalanobis_distances, degree_of_freedom)
    # pvalues = chi2.sf(square_of_mahalanobis_distances, degree_of_freedom)
    res_pvalues = ( pvalues <= threshold ) 

    chi2_crit = chi2.ppf(1 - threshold, degree_of_freedom)
    res_crit = ( square_of_mahalanobis_distances > chi2_crit )
    res_equal = np.equal(res_pvalues, res_crit)
    assert np.all(res_equal)

    return(res_pvalues, square_of_mahalanobis_distances)