#! /usr/bin/env python

'''
This code is a series of wrappers and utility functions for the pPXF code of
Cappellari & Emsellen (2004) to allow the calcium triplet to be measured using
the template fitting method of Foster et al. (2010) and Usher et al. (2018).

 

'''
 
from __future__ import print_function



import multiprocessing
import glob
import os
import datetime
import copy
import multiprocessing

import numpy as np
from scipy import constants

import ppxf
import ppxf_data
import onedspec
import interp
import normalisation
import indices

'''
A list of wavelength ranges to mask sky emission lines in region of the calcium
triplet
'''

CaT_mask = np.array([[8413, 8418], [8428, 8438], [8450, 8455], [8463, 8468],
                     [8492, 8496], [8502, 8507], [8536, 8541], [8546, 8551],
                     [8586, 8591], [8595, 8599], [8610, 8687], [8757, 8781],
                     [8823, 8839], [8848, 8852], [8865, 8870], [8882, 8888],
                     [8900, 8906], [8917, 8923], [8941, 8946], [8955, 8961]])

'''
Where to find the template files
'''
default_templates = os.environ['PPXF_TEMPLATES']

'''
Wrappers mostly for multiprocessing to work correctly
'''
def _each_sample(sample):
    return (each_sample(*sample))

def each_sample(datum, templates, vel_scale, delta_v, good_pixels, quiet, moments, degree, extra):
    
    call_ppxf(datum, templates, vel_scale, delta_v, good_pixels, quiet, moments, degree)
    if extra != None:
        extra(datum)
    return datum
        
    
def _call_ppxf(sample):
    return (call_ppxf(*sample))


'''
Wrapper around the ppxf.ppxf function
'''
def call_ppxf(datum, templates, vel_scale, delta_v, good_pixels, quiet, moments=2, degree=7, interp_func='nearest'):
    
    if moments == 2:
        priors = (datum.rv_prior, datum.sigma_prior)       
    elif moments == 4:
        priors = (datum.rv_prior, datum.sigma_prior, datum.h3_prior, datum.h4_prior)
    
    output = ppxf.ppxf(templates, datum.log_fluxes, datum.log_sigmas, vel_scale, moments=moments, mdegree=degree, degree=-1, clean=True, vsyst=delta_v, start=priors, quiet=quiet, goodpixels=good_pixels)

    datum.rv = output.sol[0]
    datum.sigma = output.sol[1]
    if moments == 4:
        datum.h3 = output.sol[2]    
        datum.h4 = output.sol[3]

    datum.log_fit_fluxes = output.bestfit
    datum.fit_wavelengths, datum.fit_fluxes = interp.logtolinear(datum.log_wavelengths, datum.log_fit_fluxes, function=interp_func, ratio=True)
    datum.rest_wavelengths = datum.fit_wavelengths / (1 + datum.rv * 1e3 / constants.c)
    
    return datum



'''
Normalise the fitted spectrum using a polynomial and measure strength of the
CaT.
Implements the method of Foster+10 and Usher+12
'''    
def measure_CaT(datum):
    if datum.normalisation_technique == 'U12':
        datum.normed_fluxes = normalisation.normalise(datum.rest_wavelengths, datum.fit_fluxes, 8, normalisation.newmask, True, .004, .010)
        
    else:
        datum.normed_fluxes = normalisation.normalise(datum.rest_wavelengths, datum.fit_fluxes, 7, normalisation.atlas_mask, True, .004, .020)
    
    datum.CaT = indices.CaT_gc(datum.rest_wavelengths, datum.normed_fluxes, normalized=True)[0]

'''
Measure the strength of the CaT using the index definition of Cenarro+01
'''
def measure_CaT_C01(datum):
    datum.CaT_C01 = indices.CaT_C01(datum.rest_wavelengths, datum.fit_fluxes)[0]
      
'''
'''  
# these DEIMOS templates should be in the templates directory
def load_CaT_templates(log_dispersion, interp_func='nearest'):
    CaT_templates = glob.glob(default_templates + 't*.fits')
    return load_fits_templates(CaT_templates, log_dispersion, interp_func)

def load_CaT_nk_templates(log_dispersion, interp_func='nearest'):
    CaT_templates = glob.glob(default_templates + 't*.fits')
    CaT_templates.remove(default_templates + 'tk.fits')
    return load_fits_templates(CaT_templates, log_dispersion, interp_func)

# you can get the templates here: http://www.iac.es/proyecto/miles/pages/stellar-libraries/miles-library.php
def load_miles_templates(log_dispersion, interp_func='nearest'):
    templates = glob.glob(default_templates + 'm*V.fits') 
    return load_fits_templates(templates, log_dispersion, interp_func)

# you can get the templates here: http://www.iac.es/proyecto/miles/pages/stellar-libraries/cat-library.php
def load_C01_templates(log_dispersion, interp_func='nearest'):
    CaT_templates = glob.glob(default_templates + 'scan*.fits') 
    return load_fits_templates(CaT_templates, log_dispersion, interp_func)

# these Indo-US templates should be in thetemplates directory
def load_indo_limited_templates(log_dispersion, interp_func='nearest'):
    CaT_templates = glob.glob(default_templates + 'indo-us*.fits')
    CaT_templates.remove(default_templates + 'indo-us_k.fits')
    return load_fits_templates(CaT_templates, log_dispersion, interp_func)

def load_indo_templates(log_dispersion, interp_func='nearest'):
    CaT_templates = glob.glob(default_templates + 'indo-us*.fits') + glob.glob(default_templates + 'extend_indo-us*.fits')
    CaT_templates.remove(default_templates + 'indo-us_k.fits')
    return load_fits_templates(CaT_templates, log_dispersion, interp_func)

def load_indo_k_templates(log_dispersion, interp_func='nearest'):
    CaT_templates = glob.glob(default_templates + 'indo-us*.fits') + glob.glob(default_templates + 'extend_indo-us*.fits')
    return load_fits_templates(CaT_templates, log_dispersion, interp_func)

'''
Load template files and redisperse them to match the observed spectrum
'''
def load_fits_templates(template_files, log_dispersion, interp_func='nearest'):
    
    #print(len(template_files))
    if len(template_files) == 0:
        raise Exception('No template files')
    log_templates = []
    for template_file in template_files:
        
        template_wavelengths, template_fluxes = onedspec.load(template_file)
        log_template_wavelengths, log_template_fluxes = interp.lineartolog(template_wavelengths, template_fluxes, function=interp_func, ratio=True, logDispersion=log_dispersion)
        log_templates.append(log_template_fluxes)
        
    log_templates = np.vstack(log_templates).T
    return log_template_wavelengths, log_templates

    
'''
Cuts input spectrum to CaT measurement wavelength range and calculates S/N of
the input spectrum
'''
def ppxf_CaT(datum, get_templates=load_CaT_templates, nsimulations=0, mask=CaT_mask, extra_function=measure_CaT, fields=['rv', 'sigma', 'CaT'], verbose=True, plot=False, interp_func='nearest'):
    zp1 = 1 + datum.rv_prior * 1e3 / constants.c
    
    lower_wavelength = 8437.0
    if not hasattr(datum, 'normalisation_technique'):
        datum.normalisation_technique = None
        
    if datum.normalisation_technique == 'U12':
        lower_wavelength = 8425.0
    
    catrange = (datum.input_wavelengths > lower_wavelength * zp1) & (datum.input_wavelengths < 8850 * zp1)
    
    datum.origial_wavelengths = datum.input_wavelengths
    datum.origial_fluxes = datum.input_fluxes
    datum.origial_sigmas = datum.input_sigmas
    
    datum.input_wavelengths = datum.input_wavelengths[catrange]
    datum.input_fluxes = datum.input_fluxes[catrange]
    datum.input_sigmas = datum.input_sigmas[catrange]
    
    s2nregion = (datum.origial_wavelengths > 8400) & (datum.origial_wavelengths < 8500)
    means2n = (datum.origial_fluxes[s2nregion] / datum.origial_sigmas[s2nregion]).mean()
    datum.s2n = means2n / (datum.origial_wavelengths[1] - datum.origial_wavelengths[0])**.5
    if verbose > 1:
        print('S/N:', datum.s2n)
    
    return run_ppxf(datum, get_templates, nsimulations, mask, extra_function, fields, moments=2, degree=7, verbose=verbose, plot=plot, interp_func=interp_func)
    
    
'''
Format input for pPXF and call pPXF
Calculate uncertanties using Monte Carlo technique
'''
def run_ppxf(datum, get_templates, nsimulations, mask=None, extra_function=None, fields=['rv', 'sigma'], moments=2, degree=7, verbose=True, plot=False, interp_func='nearest'):

    if verbose:
        print('Running ppxf:', datum.ident, datum.filename)

    if verbose > 1:
        print(datum.rv_prior, datum.sigma_prior, datum.h3_prior, datum.h4_prior)
        print(interp_func, 'interpolation')

    start = datetime.datetime.now()
    
    #Some sanity checks
    if datum.input_sigmas.min() <= 0:
        raise Exception('Non-positive sigmas')
    if datum.input_fluxes.size != datum.input_sigmas.size or datum.input_wavelengths.size != datum.input_fluxes.size:
        raise Exception('Mismatched array lengths')
    
    if datum.sigma_prior <= 0:
        raise Exception('Sigma must be positive')
    
    
    datum.nsimulations = nsimulations
    datum.mask = mask
    datum.fields = fields
    datum.log_wavelengths, datum.log_fluxes = interp.lineartolog(datum.input_wavelengths, datum.input_fluxes, function=interp_func, ratio=True)
    datum.log_wavelengths, datum.log_sigmas = interp.lineartolog(datum.input_wavelengths, datum.input_sigmas, function=interp_func, ratio=True)
    
    #Redisperse the spectrum to what pPXF expects
    logDispersion = np.log10(datum.log_wavelengths[1]) - np.log10(datum.log_wavelengths[0])
    if hasattr(get_templates, '__call__'): 
        log_template_wavelengths, log_templates = get_templates(logDispersion, interp_func)
    else:
        log_template_wavelengths, log_templates = load_fits_templates(get_templates, logDispersion, interp_func)
    delta_v = (np.log(log_template_wavelengths[0]) - np.log(datum.log_wavelengths[0])) * constants.c / 1e3
    vel_scale = logDispersion * np.log(10) * constants.c / 1e3

    #mask pixels
    regionmask = np.ones(datum.log_wavelengths.size, dtype=np.bool_)
    
    bad_pixels = np.isfinite(datum.log_sigmas)
    np.putmask(datum.log_sigmas, ~np.isfinite(datum.log_sigmas) | (datum.log_sigmas < 0), np.median(datum.log_fluxes) * 1e3)
    
    if mask is not None:
        for maskregion in mask:
            regionmask = regionmask & ~((datum.log_wavelengths > maskregion[0]) & (datum.log_wavelengths < maskregion[1]))
            
    regionmask = regionmask & bad_pixels 
    
    good_pixels = np.nonzero(regionmask)[0]
    
    if verbose > 1:
        quiet = False
    else:
        quiet = True
        
    call_ppxf(datum, log_templates, vel_scale, delta_v, good_pixels, quiet, moments, degree, interp_func)
    if extra_function != None:
        extra_function(datum)
        
    #Do the Monte Carlo uncertainty calculations
    if nsimulations > 0:
        
        samples = []
        for i in range(nsimulations):
            noise = datum.log_sigmas * np.random.normal(size=datum.log_sigmas.size)
            sample_datum = copy.copy(datum)
            sample_datum.log_fluxes = datum.log_fit_fluxes + noise
            
            samples.append((sample_datum, log_templates, vel_scale, delta_v, good_pixels, quiet, moments, degree, extra_function, interp_func))
        
        workers = min(multiprocessing.cpu_count(), 32)
        pool = multiprocessing.Pool(processes=workers)        
        
        if verbose > 1:
            print('Using', workers, 'workers')
        sample_results = pool.map(_each_sample, samples)
                
        field_results = {}
        for field in fields:
            field_results[field] = []
            
        
        for sample in sample_results:
            for field in fields:
                field_results[field].append(sample.__dict__[field])
        
        
        for field in fields:
            
            field_data = np.array(field_results[field])
            
            setattr(datum, field + '_samples', field_data)
            
            peak, lower, upper, spline = kde_interval(field_data)
            
            lower_limit = datum.__dict__[field] - lower
            upper_limit = upper - datum.__dict__[field]
            
            setattr(datum, field + '_lower', lower_limit)
            setattr(datum, field + '_upper', upper_limit)
   
            if plot:
                import matplotlib.pyplot as plt
                
                plt.figure()
                plt.title(datum.ident + ' ' + datum.filename)
                plt.hist(field_data, histtype='step', normed=True, bins=int(2 * field_data.size**0.5))
                data = np.linspace(2 * field_data.min() - field_data.mean(), 2 * field_data.max() - field_data.mean(), 512)
                plt.plot(data, spline(data))
                plt.xlabel(field)
                plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)        
                    
                    

    end = datetime.datetime.now()
    datum.runtime = (end - start).total_seconds()
    
    if verbose:
        print('Total runtime:', round(datum.runtime, 1), 's', '   Per sample:', round(datum.runtime / (nsimulations + 1), 2), 's')
        
        print_fields(datum)
        print()
    
    if plot:
        plot_fit(datum)
        
    return datum

'''
calculate n sigma interval of data
start at peak probability
integrate between two points where pdf function equals half max prob
'''
def kde_interval(data, sigma=1):
    
    p_sigma = stats.norm.cdf(sigma) - stats.norm.cdf(-sigma)
#    print p_sigma
    data = np.asanyarray(data)
    
    xs = np.linspace(2 * data.min() - data.mean(), 2 * data.max() - data.mean(), 512)
    
    # use gaussian kernel estimation and a spline to create an analytic pdf from data array
    spline = interpolate.InterpolatedUnivariateSpline(xs, stats.gaussian_kde(data)(xs))
    
    pdf_max = spline(xs).max()
    pdf = pdf_max / 2
    count = -2
    diff = 1
    roots = None
    
    while diff > 0.0025 and count > -20:
        
        shifted_spline = interpolate.InterpolatedUnivariateSpline(xs, spline(xs) - pdf)
        roots = shifted_spline.roots()
    
        if roots.size % 2 != 0:
            roots = None
            pdf += 2**-20
            continue

        p = spline.integral(roots[0], roots[-1])
                    
        if p > p_sigma:
            pdf += pdf_max * 2**count
        else:
            pdf -= pdf_max * 2**count
        
        count += -1
        diff = np.abs(p - sigma)    
    
    peak = xs[spline(xs) == pdf_max].mean()
    
    sorted_data = data[data.argsort()]
    
    return (peak, roots[0], roots[-1], spline)

'''
Print fields (and if available their confidence intervals)
'''
def print_fields(datum):
    for field in datum.fields:
        if datum.nsimulations > 0:
            print(field + ':', round(datum.__dict__[field], 4), '-' + str(round(datum.__dict__[field + '_lower'], 4)), '+' + str(round(datum.__dict__[field + '_upper'], 4)))
        else:
            print(field + ':', round(datum.__dict__[field], 4))

'''
Plot the input spectrum and the fitteded templates
'''             
def plot_fit(datum):
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 7))
    plt.subplots_adjust(left=0.05, bottom=0.08, right=0.98, top=0.96)
    plt.title(datum.ident + ' ' + datum.filename)
    
    plt.plot(datum.input_wavelengths, datum.input_fluxes, 'k-')
    plt.plot(datum.fit_wavelengths, datum.fit_fluxes, 'r-', lw=1.5)
    plt.plot(datum.input_wavelengths, datum.input_sigmas, '-', color='0.5')
    plt.plot(datum.fit_wavelengths, datum.fit_fluxes / datum.normed_fluxes, 'b:')
    
    mean_flux = datum.fit_fluxes.mean()
    if datum.mask is not None:
        plt.hlines(np.ones(datum.mask.shape[0]) * mean_flux / 8, datum.mask[:,0], datum.mask[:,1], color='y', lw=2)
    
    delta_wavelengths = datum.input_wavelengths.max() - datum.input_wavelengths.min()
    plt.xlim(datum.input_wavelengths.min() - 0.02 * delta_wavelengths, datum.input_wavelengths.max() + 0.02 * delta_wavelengths)    
    plt.ylim(-mean_flux / 20, max(mean_flux + 3 * datum.fit_fluxes.std(), datum.input_fluxes.mean() + 2 * datum.input_fluxes.std()))
    plt.xlabel(u'Wavelength (\u00C5)')
    
    
