#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:28:19 2023

@author: user1

"""

import os
os.environ['CRDS_PATH'] ='/Users/c24050258/crds_cache'
os.environ['CRDS_SERVER_URL'] ='https://jwst-crds.stsci.edu'
# #this must come BEFORE jwst is imported

import asdf
import copy
import shutil
import numpy as np
import requests
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch
import matplotlib.pyplot as plt
import matplotlib as mpl

# jwst imports 
import jwst
print(jwst.__version__)

from jwst.pipeline import calwebb_detector1

# Individual steps that make up calwebb_detector1
from jwst.dq_init import DQInitStep
from jwst.saturation import SaturationStep
from jwst.superbias import SuperBiasStep
from jwst.ipc import IPCStep                                                                                    
from jwst.refpix import RefPixStep                                                                
from jwst.linearity import LinearityStep
from jwst.persistence import PersistenceStep
from jwst.dark_current import DarkCurrentStep
from jwst.jump import JumpStep
from jwst.ramp_fitting import RampFitStep
from jwst import datamodels

from jwst.datamodels import dqflags

# miri specific steps
from jwst.rscd import RscdStep
from jwst.firstframe import FirstFrameStep
from jwst.lastframe import LastFrameStep
from jwst.reset import ResetStep
from jwst.gain_scale import GainScaleStep
from jwst.group_scale import GroupScaleStep

# import pipeline_lib

output_dir = './output'
if not os.path.exists(output_dir ): 
    os.makedirs(output_dir )
    
from jwst.stpipe import Step 

for i in range(1,2):

    file = '/Users/c24050258/Library/CloudStorage/OneDrive-CardiffUniversity/Projects/JWST_Test_Code/Data/jw01366004001_04101_00001-seg00'+str(i)+'_nrs1/jw01366004001_04101_00001-seg00'+str(i)+'_nrs1_uncal.fits'
         
    result = file 
    
    step = GroupScaleStep()
    result = step.run(result)
    
     
    step = DQInitStep()
    result = step.run(result)
            
    step = SaturationStep()
    #     if channel=='grating':
    #         step.n_pix_grow_sat =1 
    #     if channel == 'prism':
    #         step.n_pix_grow_sat = sp_factor  # important.... 
    result = step.run(result)
     
    step = SuperBiasStep()
    # if channel == 'grating':
    #     if nrs ==1 :
    #         step.override_superbias ='/Users/user1/crds_cache/references/jwst/nirspec/jwst_nirspec_superbias_0427.fits'
    # step.output_dir = output_dir
    # step.save_results = True
    result = step.run(result)
          
    step = RefPixStep()
    # step.output_dir = output_dir
    # step.save_results = True
    result = step.run(result)

    for j in range(result.data.shape[0]):
        first_pix = result.data[j,:,0:5,:]
        last_pix = result.data[j,:,-5:,:]
        bkg_stack = np.hstack((first_pix,last_pix))
        bkg_med = np.nanmedian(bkg_stack, axis=1)
        bkg_3d = np.expand_dims(bkg_med, axis=1)
        bkg = np.repeat(bkg_3d, result.shape[2], axis=1)
        result.data[j] = result.data[j] - bkg
    
    # if channel == 'prism':
    #     step = pipeline_lib.CustomBkg()
    #     result = step.run(result)
    # elif channel == 'grating':
    #     step = pipeline_lib.CustomBkgStage1_grating()
    #     result = step.run(result, nrs)
            
    step = LinearityStep()
    # step.output_dir = output_dir
    result = step.run(result)
          
    step = DarkCurrentStep()
    result = step.run(result)
     
    step = JumpStep()
    step.rejection_threshold  = 15
    print ("step.rejection_threshold", step.rejection_threshold)
    result = step.run(result)           
    
            
    step = RampFitStep()
    # step.save_results = True
    # step.save_opt = False
    # step.suppress_one_group = True # default
    # if channel == 'prism':
    #     step.suppress_one_group = False # allows slopes to be obtained from gp 1 in sat ramps
    result = step.run(result)[1]
      
         
    step = GainScaleStep()
    result = step.run(result)
    
    print (result.data.shape)
    plt.figure('img 1')
    plt.imshow(result.data[0], aspect='auto')
    plt.show()
    
    
    # import re
    # idx1 = [m.start() for m in re.finditer('/', file)][-1]+1
    # idx2  =  file.find('.fits')
    # file_name = file[idx1:idx2]
    file_name = file.replace('uncal', 'rateints')
    # file_path = './fits_files/%s'%(channel)
    # if not os.path.exists(file_path):
    #     os.makedirs(file_path)
    # result.save('%s/%s_%s.fits'%(file_path,file_name, tag)) 
     
    result.save('%s.fits'%(file_name)) 