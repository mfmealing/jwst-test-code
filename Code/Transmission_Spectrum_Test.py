import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
from uncertainties import ufloat
 
matplotlib.style.use('classic')

 
import pylightcurve as plc
from astropy.io import fits
import emcee
import corner
# import pandas as pd
 

from lmfit import Model as lmfit_Model

def transit_model(t, rat, t0, gamma0, gamma1, per, ars, inc, w, ecc, a, b, ldc_type='quad'):

    lc = plc.transit([gamma0, gamma1], rat, per, ars, ecc, inc, w, t0, t, method=ldc_type)
    syst = (a*t) + b
 
    lc = lc * syst
    return lc

seg_list = ['001', '002', '003', '004']

plt.figure('segments')
for seg in seg_list: 

    f='/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity//Projects/JWST_Test_Code/Data/jw01366004001_04101_00001-seg%s_nrs1/jw01366004001_04101_00001-seg%s_nrs1_1Dspec_box_extract.fits.fits'%(seg, seg)
    
    hdul = fits.open(f)
    int_times = hdul[1].data
    slc = hdul[2].data
    wav = hdul[3].data
    var = (hdul[4].data)**2
    
    bjd = int_times['int_mid_BJD_TDB']
    mjd= int_times['int_mid_MJD_UTC']
    
    idx = np.argwhere((wav<1.0 ) | (wav>2.0)).T[0]
    wlc = np.nansum(slc[:, idx], axis=1)
    
    if seg == seg_list[0]:
        slc_stack =slc
        var_stack =var
        bjd_stack = bjd
    else:
        slc_stack = np.vstack((slc_stack, slc))
        var_stack = np.vstack((var_stack, slc))
        bjd_stack = np.hstack((bjd_stack, bjd))

    
    plt.plot(bjd, wlc)
    
wlc = np.nansum(slc_stack[:, idx], axis=1)
slc = slc_stack
var = var_stack
bjd = bjd_stack
plt.figure('wlc')
plt.plot(bjd, wlc, '.')


fixed_vals = np.loadtxt('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/JWST_Test_Code/Data/lm_fit_fixed_vals.csv', delimiter=',')

t= bjd-bjd[0]
lm_rat = (21228/1e6)**0.5
lm_t0 = fixed_vals[0]
lm_gamma0 = fixed_vals[1]
lm_gamma1 = fixed_vals[2]
lm_per = 4.0552941
lm_ars = fixed_vals[3]
lm_inc = fixed_vals[4]
lm_w = 90
lm_ecc = 0
lm_a = (wlc[-1]-wlc[0])/(t[-1]-t[0])
lm_b = wlc[0]

gmodel = lmfit_Model(transit_model)

lm_params = gmodel.make_params()
lm_params.add('rat', value=lm_rat,vary=True)
lm_params.add('t0', value=lm_t0,  vary=False)
lm_params.add('gamma0', value=lm_gamma0,  vary=False)
lm_params.add('gamma1', value=lm_gamma1,  vary=False)
lm_params.add('per', value=lm_per, vary=False)
lm_params.add('ars', value=lm_ars, vary=False)
lm_params.add('inc', value=lm_inc, vary=False)
lm_params.add('w', value=lm_w,  vary=False)
lm_params.add('ecc', value=lm_ecc, vary=False)
lm_params.add('a', value=lm_a, vary=True)
lm_params.add('b', value=lm_b, vary=True)


wav_bin = 10
count = 0
slc_new = []
var_new = []
wav_avg = []
rat = []
rat_err = []


for i in range(1+int(len(slc[1])/wav_bin)):
    slc_new.append(np.nansum(slc[:,count:(count+wav_bin)],axis=1))
    var_new.append(np.nansum(var[:,count:(count+wav_bin)],axis=1))
    wav_avg.append(np.mean(wav[count:(count+wav_bin)]))
    count += wav_bin
    
    result = gmodel.fit(slc_new[i], lm_params, t=t, ldc_type = 'quad')
    #print(result.params['rat'].value)
    rat.append(result.params['rat'].value)
    rat_err.append(result.params['rat'].stderr)

# for j in range(len(slc_new)):
#     plt.figure('slc all')
#     plt.plot(bjd, slc_new[j], '.')

rprs2 = []
rprs2_err = []

for k in range(len(rat)):
    x = ufloat(rat[k],rat_err[k])
    x = x**2
    rprs2.append(x.nominal_value)
    rprs2_err.append(x.std_dev)

# final_data = (wav_avg, rprs2, rprs2_err)
# np.savetxt('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity//Projects/JWST_Test_Code/Data/final_spectrum_data.csv', final_data, delimiter=',')

result_old = np.loadtxt('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity//Projects/JWST_Test_Code/Data/final_spectrum_data_OLD.csv', delimiter=',')
result_final = np.loadtxt('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity//Projects/JWST_Test_Code/Data/final_spectrum_data.csv', delimiter=',')

wav_old = result_old[0]
r_old = result_old[1]
r_err_old = result_old[2]
wav_final = result_final[0]
r_final = result_final[1]
r_err_final = result_final[2]

# plt.figure('ratio spectrum')
# plt.errorbar(wav_avg, rat, rat_err, fmt='o')

plt.figure('transmission spectrum')
plt.errorbar(wav_old, r_old, r_err_old, fmt='ro', label='old data')
plt.errorbar(wav_final, r_final, r_err_final, fmt='bo', label='new data')
plt.legend(loc='lower center', numpoints=1)






fixed_vals_mcmc = np.loadtxt('/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/JWST_Test_Code/Data/mcmc_fit_fixed_vals.csv', delimiter=',')

theta = (lm_rat, lm_a, lm_b)
params = fixed_vals_mcmc
n_walkers = 500  # Change back to 500 after testing
n_dim = 3
n_iter = 2000


def model(theta, params=params, per=lm_per, w=lm_w, ecc=lm_ecc, ldc_type='quad'):
    rat, a, b = theta
    t0, gamma0, gamma1, ars, inc = params
    lc = plc.transit([gamma0, gamma1], rat, per, ars, ecc, inc, w, t0, t, method=ldc_type)
    syst = (a*t) + b
    lc = lc * syst
    
    return lc

def lnlike(theta, x, y, y_err):
    return -0.5 * np.sum(((y - model(theta))/y_err) ** 2)

def lnprior(theta):
    rat, a, b = theta
    if 0.1 < rat < 0.3 and -1e6 < a < -3e6 and 4e8 < b < 7e8:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, y_err):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, y_err)

def mcmc(p0, n_walkers, n_iter, n_dim, lnprob, data):
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, args=data)
    p0, _, _ = sampler.run_mcmc(p0, 1000)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(p0, n_iter)
    return sampler, pos, prob, state


mcmc_rat = []
mcmc_rat_err = []

for j in range(1+int(len(slc[1])/wav_bin)):
    data = (t, slc_new[j], var_new[j])
    initial = np.array([lm_rat, lm_a, lm_b])
    p0 = [np.array(initial) + 1e-7 * np.random.randn(n_dim) for i in range(n_walkers)]

    sampler, pos, prob, state = mcmc(p0, n_walkers, n_iter, n_dim, lnprob, data)
    samples = sampler.flatchain
    
    vals = np.percentile(samples[0,0], [16, 50, 84])
    # print(vals)
    err = np.diff(vals)
    mcmc_rat.append(vals[1])
    mcmc_rat_err.append(np.mean(err))

    
labels = ['rat', 'a', 'b']
fig = corner.corner(samples, show_titles=True, labels=labels, smooth=True, quantiles=[0.16, 0.5, 0.84])

plt.figure('mcmc ratio spectrum')
# plt.plot(wav_avg, mcmc_rat, '.')
plt.errorbar(wav_avg, mcmc_rat, mcmc_rat_err, fmt='o')

mcmc_rprs2 = []
mcmc_rprs2_err = []

for l in range(len(mcmc_rat)):
    x = ufloat(mcmc_rat[l],mcmc_rat_err[l])
    x = x**2
    mcmc_rprs2.append(x.nominal_value)
    mcmc_rprs2_err.append(x.std_dev)

plt.figure('mcmc transmission spectrum')
plt.errorbar(wav_avg, mcmc_rprs2, mcmc_rprs2_err, fmt='o')