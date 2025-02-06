import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
 
matplotlib.style.use('classic')

 
import pylightcurve as plc
from astropy.io import fits
import emcee
import corner
# import pandas as pd
 

from lmfit import Model as lmfit_Model

def transit_model(t, rat, t0, gamma0, gamma1, per, ars, inc, w, ecc, a, b, c, ldc_type='quad'):

    lc = plc.transit([gamma0, gamma1], rat, per, ars, ecc, inc, w, t0, t, method=ldc_type)
    syst = (a*t**2) + (b*t) + c
    
    # syst = A*np.exp(-B*t)+ a*t + b
 
    lc = lc * syst
    return lc

def transit_model_exp(t, rat, t0, gamma0, gamma1, per, ars, inc, w, ecc, A, B, ldc_type='quad'):

    lc = plc.transit([gamma0, gamma1], rat, per, ars, ecc, inc, w, t0, t, method=ldc_type)
    syst = A * np.exp(-t/B)
 
    lc = lc * syst
    return lc

seg_list = ['001', '002', '003', '004']

plt.figure('segments')
for seg in seg_list: 

    f='/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/JWST_Test_Code/Data/jw01366004001_04101_00001-seg%s_nrs1/jw01366004001_04101_00001-seg%s_nrs1_1Dspec_box_extract.fits.fits'%(seg, seg)
    
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

# plt.figure('integration')
# plt.plot(slc[10000])

# plt.figure('slc')
# plt.plot(bjd, slc[:,200], '.')

# plt.figure('slc bin')
# plt.plot(bjd, np.nansum(slc[:,195:206], axis=1), '.')

# plt.figure('spectrum vs wavelength')
# plt.plot(wav, slc[10000])


# 4. time-bin
# ============================================================================= 
plt.figure('wlc after time bin')
time_bin=40
idx  =  np.arange(0, slc.shape[0], time_bin)
bjd =   (np.add.reduceat(bjd, idx)/  time_bin) [:-1]
slc  =  np.add.reduceat(slc, idx, axis=0)[:-1]
var  =  np.add.reduceat(var, idx, axis=0)[:-1]
idx = np.argwhere((wav<1.0 ) | (wav>2.0)).T[0]
wlc  =  np.nansum(slc[:,idx],axis=1)
wlc_var  =  np.nansum(var[:,idx],axis=1)
print ('time_step (s): ', np.diff(bjd)[0]*24*60*60)   
plt.errorbar(bjd, wlc, wlc_var**0.5, fmt='ro')


# =============================================================================
# initial guess
# =============================================================================
t= bjd-bjd[0]
t_idx = np.argwhere((t>0.05) & (wlc>5.2557e8)).T[0]
t = t[t_idx]
wlc = wlc[t_idx]
wlc_var = wlc_var[t_idx]

            
lm_rat = (21228/1e6)**0.5
lm_t0 = 59770.83991 - bjd[0]
lm_gamma0 = 0.2
lm_gamma1 = 0.2
lm_per = 4.0552941
lm_ars = 11.308276
lm_inc = 87.637683
lm_w = 90
lm_ecc = 0
lm_a = 1
lm_b = (wlc[-1]-wlc[0])/(t[-1]-t[0])
lm_c = wlc[0]
lm_A = wlc[0]
lm_B = 200
                
initial_guess_lin = transit_model(t, lm_rat, lm_t0, lm_gamma0 , lm_gamma1,
                                lm_per, lm_ars, lm_inc, 
                                lm_w, lm_ecc, 0, lm_b, lm_c)

# initial_guess_quad = transit_model(t, lm_rat, lm_t0, lm_gamma0 , lm_gamma1,
#                                 lm_per, lm_ars, lm_inc, 
#                                 lm_w, lm_ecc, lm_a, lm_b, lm_c)

# initial_guess_exp = transit_model_exp(t, lm_rat, lm_t0, lm_gamma0 , lm_gamma1,
#                                 lm_per, lm_ars, lm_inc, 
#                                 lm_w, lm_ecc, lm_A, lm_B)



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,5))
ax1.plot(t, wlc, 'bo')
ax1.plot(t, initial_guess_lin, 'g-', linewidth=3)
ax1.set_title('intial guess: linear')
# ax2.plot(t, wlc, 'bo')
# ax2.plot(t, initial_guess_quad, 'g-', linewidth=3)
# ax2.set_title('intial guess: quadratic')
# ax3.plot(t, wlc, 'bo')
# ax3.plot(t, initial_guess_exp, 'g-', linewidth=3)
# ax3.set_title('intial guess: exponential')
# plt.tight_layout()

# =============================================================================
# fit with linear systematics
# =============================================================================

gmodel_lin = lmfit_Model(transit_model)

lm_params_lin = gmodel_lin.make_params()
lm_params_lin.add('rat', value=lm_rat,vary=True)
lm_params_lin.add('t0', value=lm_t0,  vary=True)
lm_params_lin.add('gamma0', value=lm_gamma0,  vary=True, min =0, max=1)
lm_params_lin.add('gamma1', value=lm_gamma1,  vary=True, min=0, max=1)
lm_params_lin.add('per', value=lm_per, vary=False)
lm_params_lin.add('ars', value=lm_ars, vary=True)
lm_params_lin.add('inc', value=lm_inc, vary=True)
lm_params_lin.add('w', value=lm_w,  vary=False)
lm_params_lin.add('ecc', value=lm_ecc, vary=False)
lm_params_lin.add('a', value = 0, vary=False)
lm_params_lin.add('b', value=lm_b, vary=True)
lm_params_lin.add('c', value=lm_c, vary=True)
     
result_lin = gmodel_lin.fit(wlc, lm_params_lin, t=t, ldc_type = 'quad', weights=(1/wlc_var**(1/2)))
print (result_lin.fit_report())

model_fit_lin = transit_model(t, result_lin.params['rat'].value, 
                         result_lin.params['t0'], result_lin.params['gamma0'], result_lin.params['gamma1'], lm_per, 
                         result_lin.params['ars'], result_lin.params['inc'], lm_w, lm_ecc,
                         result_lin.params['a'].value, result_lin.params['b'].value, result_lin.params['c'].value)


# # =============================================================================
# # fit with quadratic systematics
# # =============================================================================

# gmodel_quad = lmfit_Model(transit_model)

# lm_params_quad = gmodel_quad.make_params()
# lm_params_quad.add('rat', value=lm_rat,vary=True)
# lm_params_quad.add('t0', value=lm_t0,  vary=True)
# lm_params_quad.add('gamma0', value=lm_gamma0,  vary=True, min =0, max=1)
# lm_params_quad.add('gamma1', value=lm_gamma1,  vary=True, min=0, max=1)
# lm_params_quad.add('per', value=lm_per, vary=False)
# lm_params_quad.add('ars', value=lm_ars, vary=True)
# lm_params_quad.add('inc', value=lm_inc, vary=True)
# lm_params_quad.add('w', value=lm_w,  vary=False)
# lm_params_quad.add('ecc', value=lm_ecc, vary=False)
# lm_params_quad.add('a', value=lm_a, vary=True)
# lm_params_quad.add('b', value=lm_b, vary=True)
# lm_params_quad.add('c', value=lm_c, vary=True)

# result_quad = gmodel_quad.fit(wlc, lm_params_quad, t=t, ldc_type = 'quad', weights=(1/wlc_var**(1/2)))
# print (result_quad.fit_report())

# model_fit_quad = transit_model(t, result_quad.params['rat'].value, 
#                          result_quad.params['t0'], result_quad.params['gamma0'], result_quad.params['gamma1'], lm_per, 
#                          result_quad.params['ars'], result_quad.params['inc'], lm_w, lm_ecc,
#                          result_quad.params['a'].value, result_quad.params['b'].value, result_quad.params['c'].value)


# # =============================================================================
# # fit with exponential systematics
# # =============================================================================

# gmodel_exp = lmfit_Model(transit_model_exp)

# lm_params_exp = gmodel_exp.make_params()
# lm_params_exp.add('rat', value=lm_rat,vary=True)
# lm_params_exp.add('t0', value=lm_t0,  vary=True)
# lm_params_exp.add('gamma0', value=lm_gamma0,  vary=True, min =0, max=1)
# lm_params_exp.add('gamma1', value=lm_gamma1,  vary=True, min=0, max=1)
# lm_params_exp.add('per', value=lm_per, vary=False)
# lm_params_exp.add('ars', value=lm_ars, vary=True)
# lm_params_exp.add('inc', value=lm_inc, vary=True)
# lm_params_exp.add('w', value=lm_w,  vary=False)
# lm_params_exp.add('ecc', value=lm_ecc, vary=False)
# lm_params_exp.add('A', value=lm_A, vary=True)
# lm_params_exp.add('B', value=lm_B, vary=True)

# result_exp = gmodel_exp.fit(wlc, lm_params_exp, t=t, ldc_type = 'quad', weights=(1/wlc_var**(1/2)))
# print (result_exp.fit_report())

# model_fit_exp = transit_model_exp(t, result_exp.params['rat'].value, 
#                          result_exp.params['t0'], result_exp.params['gamma0'], result_exp.params['gamma1'], lm_per, 
#                          result_exp.params['ars'], result_exp.params['inc'], lm_w, lm_ecc,
#                          result_exp.params['A'].value, result_exp.params['B'].value)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,5))
ax1.plot(t, wlc, 'bo')
ax1.plot(t, model_fit_lin, 'r-', linewidth = 3)
ax1.set_title('lm_fit linear')
# ax2.plot(t, wlc, 'bo')
# ax2.plot(t, model_fit_quad, 'r-', linewidth = 3)
# ax2.set_title('lm_fit quadratic')
# ax3.plot(t, wlc, 'bo')
# ax3.plot(t, model_fit_exp, 'r-', linewidth = 3)
# ax3.set_title('lm_fit exponential')
# plt.tight_layout()


# =============================================================================
# residuals and best fit models
# =============================================================================

res_lin = model_fit_lin - wlc
# res_quad = model_fit_quad - wlc
# res_exp = model_fit_exp - wlc
# print(np.std(res_lin), np.std(res_quad), np.std(res_exp))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,5))
ax1.plot(t, res_lin, '.')
ax1.grid(True)
ax1.set_title('linear residuals')
# ax2.plot(t, res_quad, '.')
# ax2.grid(True)
# ax2.set_title('quadratic residuals')
# ax3.plot(t, res_exp, '.')
# ax3.grid(True)
# ax3.set_title('exponential residuals')
# plt.tight_layout()

var_lin = np.var(res_lin)
# var_quad = np.var(res_quad)
# var_exp = np.var(res_exp)

chi_lin = np.nansum(res_lin**2/var_lin)
# chi_quad = np.nansum(res_quad**2/var_quad)
# chi_exp = np.nansum(res_exp**2/var_exp)
# print(chi_lin, chi_quad, chi_exp)

reduce_chi_lin = chi_lin/len(wlc)
# reduce_chi_quad = chi_quad/len(wlc)
# reduce_chi_exp = chi_exp/len(wlc)
# print(reduce_chi_lin, reduce_chi_quad, reduce_chi_exp)






# =============================================================================
# MCMC
# =============================================================================

theta = (lm_rat, lm_t0, lm_gamma0 , lm_gamma1, lm_ars, lm_inc, lm_b, lm_c)
n_walkers = 64
n_dim = 8
n_iter = 2000

lm_a = 0
lm_per = 4.0552941
lm_w = 90
lm_ecc = 0

def model(theta, a=lm_a, per=lm_per, w=lm_w, ecc=lm_ecc, ldc_type='quad'):
    rat, t0, gamma0, gamma1, ars, inc, b, c = theta
    lc = plc.transit([gamma0, gamma1], rat, per, ars, ecc, inc, w, t0, t, method=ldc_type)
    syst = (a*t**2) + (b*t) + c
 
    lc = lc * syst
    return lc

def lnlike(theta, x, y, y_err):
    return -0.5 * np.sum(((y - model(theta))/y_err) ** 2)

def lnprior(theta):
    rat, t0, gamma0, gamma1, ars, inc, b, c = theta
    if 0.1 < rat < 0.2 and 0.1 < t0 < 0.3 and 0.0 < gamma0 < 1.0 and 0.0 < gamma1 < 1.0 and 5.0 < ars < 15.0 and 80.0 < inc < 90.0 and -1e6 < b < -3e6 and 4e8 < c < 7e8:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, y_err):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, y_err)

data = (t, wlc, wlc_var)
initial = np.array([lm_rat, lm_t0, lm_gamma0, lm_gamma1, lm_ars, lm_inc, lm_b, lm_c])
p0 = [np.array(initial) + 1e-4 * np.random.randn(n_dim) for i in range(n_walkers)]

sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, args=(t, wlc, wlc_var))
p0, _, _ = sampler.run_mcmc(p0, 1000, progress=True, tune=True)
sampler.reset()
pos, prob, state = sampler.run_mcmc(p0, n_iter, progress=True, tune=True)
samples = sampler.flatchain


theta_max  = samples[np.argmax(sampler.flatlnprobability)]
best_fit_model = model(theta_max)
plt.figure('mcmc fit')
plt.plot(t, wlc, 'bo')
plt.plot(t, best_fit_model, 'r-')
plt.show()

labels = ['rat', 't0', 'gamma0', 'gamma1', 'ars', 'inc', 'b', 'c']
fig = corner.corner(samples, show_titles=True, title_fmt='.2f', labels=labels, smooth=True, quantiles=[0.16, 0.5, 0.84])

# final_vals = []
# for i in range(n_dim):
#     vals = np.percentile(samples[:, i], [16, 50, 84])
#     final_vals.append(vals[1])