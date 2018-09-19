#! /usr/bin/env python

'''

Code to measure the strength of the calcium triplet (CaT) using the template
fitting method of Foster et al. (2010) and Usher et al. (2018) using input from
the command line

Will accept pickled ppxf_data objects (as .ppxf), fits spectra and ascii
spectra as input

'''

from __future__ import print_function

import argparse
import pickle

import ppxf_lib
import ppxf_data



parser = argparse.ArgumentParser(description='Run pPXF on a single spectrum and optionally use Monte Carlo simulations to estimate the uncertainty on the fit')
parser.add_argument('input', help='Input file. If file ends in .fits, the code assumes it is a fits file. If it ends in .ppxf, it assumes it is a pickled ppxf_data object. Otherwise, the code assumes that it is ascii')
parser.add_argument('errors', nargs='?', help='Error file. Only used if input file is a fits file')

error_type = parser.add_mutually_exclusive_group()
error_type.add_argument('--ivars', action='store_true', help='Uncertainty is stored as inverse variances')
error_type.add_argument('--varis', action='store_true', help='Uncertainty is stored as variances')


parser.add_argument('-v', '--rv', type=float, default=0, help='Radial velocity prior')
parser.add_argument('-s', '--sigma', type=float, default=10, help='Velocity dispersion prior')
parser.add_argument('--h3', type=float, default=0, help='h3 prior')
parser.add_argument('--h4', type=float, default=0, help='h4 prior')

parser.add_argument('--order', type=float, default=7, help='Order of the continuum polynomial')
parser.add_argument('--moments', type=float, default=2, help='Velocity moments to fit')

parser.add_argument('-N', '--num-simulations', type=int, default=0, help='Number of Monte Carlo simulations')

verboseness = parser.add_mutually_exclusive_group()
verboseness.add_argument('--quiet', action='store_true', help='Suppress output')
verboseness.add_argument('--loud', action='store_true', help='Provide extra output')

extras = parser.add_mutually_exclusive_group()
extras.add_argument('--CaT', action='store_true', help='Measure CaT using Usher+12 technique')
extras.add_argument('--CaT_C01', action='store_true', help='Measure CaT using Cennaro+01 definition')
extras.add_argument('--kinematics', action='store_true', help='Just measure the kinematics')

parser.add_argument('--no-mask', action='store_true', help='Don\'t mask wavelength ranges with emission lines')


parser.add_argument('--templates', nargs='+', help='Template files.\nUse DEIMOS for the DEIMOS templates, Cenarro01 for the Cenarro 2001 library and INDO for the INDO-US templates')
parser.add_argument('--save', action='store_true', help='Save output as pickled ppxf_data object')
parser.add_argument('-n', '--name', default='', help='Object name')
parser.add_argument('-p', '--plot', action='store_true', help='Plot fitted spectra and Monte Carlo results')

parser.add_argument('--normalisation', help='Use different normalisation parameters\n Use U12 for parameters in Usher et al. (2012)')


if __name__ == "__main__":
    args = parser.parse_args()

    if args.quiet:
        verbose = 0
    elif args.loud:
        verbose = 2
    else:
        verbose = 1

    if args.normalisation is not None:
        print('Using', args.normalisation, 'normalisation')
        
#    print(args.templates)
                
    if args.templates == None:
        template_str = 'Using standard templates'
    elif args.templates[0] in ['DEIMOS', 'deimos']:
        template_str = 'Using the DEIMOS templates'
        templates = ppxf_lib.load_CaT_templates
    elif args.templates[0] in ['Cenarro01', 'C01', 'cenarro', 'Cenarro']:
        template_str = 'Using the Cenarro 2001 templates'
        templates = ppxf_lib.load_C01_templates
    elif args.templates[0] in ['indo', 'INDO']:
        template_str = 'Using the INDO-US templates'
        templates = ppxf_lib.load_indo_templates
    elif args.templates[0] in ['limited']:
        template_str = 'Using the limited INDO-US templates'
        templates = ppxf_lib.load_indo_limited_templates
        
        
    elif args.templates != None:
        template_str = 'Using user supplied templates'
        templates = args.templates

    if verbose:
        print(template_str)

    #if input file is a pickled ppxf_data object
    if args.input[-5:] == '.ppxf':
        
        input_datum = pickle.load(open(args.input))
        output_name = args.input[:-5]
        args.save = True
        input_datum.normalisation_technique = args.normalisation
    
    #if input file is a fits file    
    elif args.input[-5:] == '.fits':
        
        input_datum = ppxf_data.create_from_fits(args.input, args.errors, ident=args.name, ivars=args.ivars, varis=args.varis, rv_prior=args.rv, sigma_prior=args.sigma, h3_prior=args.h3, h4_prior=args.h4)
        input_datum.normalisation_technique = args.normalisation
        output_name = args.input[:-5]
    
    #otherwise assume ascii    
    else:
               
        input_datum = ppxf_data.create_from_ascii(args.input, ident=args.name, ivars=args.ivars, varis=args.varis, rv_prior=args.rv, sigma_prior=args.sigma, h3_prior=args.h3, h4_prior=args.h4)
        input_datum.normalisation_technique = args.normalisation
        output_name = args.input

    if args.no_mask:
        mask = None
    else:
        mask = ppxf_lib.CaT_mask        
    
        
    if args.templates == None:
        templates = ppxf_lib.load_CaT_templates

    #actually do the fit and measurement
    output_datum = ppxf_lib.ppxf_CaT(input_datum, templates, nsimulations=args.num_simulations, verbose=verbose, plot=args.plot, mask=mask)
        

    #only save results if prompted
    if args.save:
        pickle.dump(output_datum, open(output_name + '.ppxfout', 'w'))
        
    if args.plot:
        import matplotlib.pyplot as plt
        plt.show()
            
    
            
            
        
        
        
    
        
        
        
        
    
    

