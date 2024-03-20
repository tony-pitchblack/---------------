import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

def _fit_ridge(
            model, alpha,
            cov_type = "nonrobust",
            cov_kwds=None,
            use_t = None,
            **kwargs,
            ):
        """
        Fit a linear model using ridge regression.

        Parameters
        ----------
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.

        Notes
        -----
        Equivalent to fit_regularized with L1_wt = 0 (but implemented
        more efficiently).
        """

        model.fit()
        u, s, vt = np.linalg.svd(model.exog, 0)
        v = vt.T
        q = np.dot(u.T, model.endog) * s
        s2 = s * s
        if np.isscalar(alpha):
            sd = s2 + alpha * model.nobs
            params = q / sd
            params = np.dot(v, params)
        else:
            alpha = np.asarray(alpha)
            vtav = model.nobs * np.dot(vt, alpha[:, None] * v)
            d = np.diag(vtav) + s2
            np.fill_diagonal(vtav, d)
            r = np.linalg.solve(vtav, q)
            params = np.dot(v, r)

        lfit = sm.regression.linear_model.OLSResults(
            model, params,
            normalized_cov_params=model.normalized_cov_params,
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        
        return sm.regression.linear_model.RegressionResultsWrapper(lfit)

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
    model_ar1 = sm.OLS(ystar,xstar)
    return(model_ar1)

# cochrane-orcutt / prais-winsten iterative procedure
# default to cochrane-orcutt (drop1=True)
def OLSAR1(model, n_iter=np.inf, drop1=False):
    x = model.exog
    y = model.endog
    res = model.fit()
    e = res.resid
    e1 = e[:-1]; e0 = e[1:]
    rho0 = np.dot(e1,e0)/np.dot(e1,e1)
    rdiff = 1.0
    i = 0
    while(rdiff>1.0e-5 and i < n_iter):
        model1 = ols_ar1(model,rho0,drop1)
        e = model1.fit().resid
        e1 = e[:-1]; e0 = e[1:]
        rho1 = np.dot(e1,e0)/np.dot(e1,e1)
        rdiff = np.sqrt((rho1-rho0)**2)
        rho0 = rho1
        i += 1
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

def get_whites_regr(regression, k_most_corr=None, full=False):
    if full and k_most_corr:
        raise ValueError('either full or k_most_corr must be supplied')

    result_OLS = regression.fit()
    e = result_OLS.resid
    X = pd.DataFrame(data=regression.exog, columns=regression.exog_names,
                        index=e.index).iloc[:, 1:]
    X_resid = X.join(pd.DataFrame({"e": result_OLS.resid}))
    kendall_corr = X_resid.corr(method='kendall')
    most_corr_idx = kendall_corr.iloc[:-1, [-1]].sort_values(by='e', ascending=False,
                                            key=lambda series: series.abs()).index
    
    if full:
        X_white = X.copy()
        for i, colname_i in enumerate(X):
            for colname_j in X.columns[i:]:
                colname_ij = colname_i + ' * ' + colname_j 
                X_white[colname_ij] = X[colname_i] * X[colname_j]
    else:
        X_white = X[most_corr_idx[:k_most_corr]] ** 2
        X_white = X_white.rename(columns={col: col+'^2' for col in X_white.columns})
        # X_white = X.merge(X_white, left_index=True, right_index=True)
    X_white = sm.add_constant(X_white)
    whites_resid_regr = sm.OLS(np.log(e ** 2), X_white, hasconst=True)
    # w_res = whites_resid_regr.fit()
    # print(w_res.summary())

    sigma_sq_log = whites_resid_regr.fit().predict()
    weights = 1/np.exp(sigma_sq_log)
    whites_regr = sm.WLS(regression.endog, regression.exog,
                            weights=weights, hasconst=True)

    return(whites_regr, whites_resid_regr.exog_names)