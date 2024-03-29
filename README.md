# measure_CaT
Code to measure the strength of the calcium triplet (CaT) spectral feature by fitting stellar templates using the method described in [Foster et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010AJ....139.1566F/abstract) and [Usher et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1275U/abstract).

As described in Usher et al. (2018), the code measures the strength of the CaT by fitting a linear combination of stellar templates using the pPXF code of [Cappellari & Emsellem (2004)](https://ui.adsabs.harvard.edu/#abs/2004PASP..116..138C), normalising the fitted templates using a polynomial fit to regions of low line opacity, and then measure the index strength on the normalised templates. Some of the code was written to be a more general wrapper around pPXF.

Any publication resulting from the use of this software should cite [Usher et al., 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1275U/abstract) and [Cappellari & Emsellem (2004)](https://ui.adsabs.harvard.edu/#abs/2004PASP..116..138C). Although modification of this code is encouraged, any publication resulting from the modified code should state that the code has been modified and explain how the code has been modified.

### Setup
Besides pPXF (available from http://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf), this code requires Numpy, SciPy and optionally Matplotlib to run. Running this code requires no installation - just put `src/` in your Python path and set the `PPXF_TEMPLATE` environment variable to the location of the `templates/` directory.

### Using measure_CaT
The code can be used in two different ways. From the command line run `measure_CaT.py` <filename>. This script can handle spectra in FITS format, ascii or as a pickled `ppxf_data` object.

From within a Python script

```
input_datum = ppxf_data.ppxf_data(wavelengths, fluxes, sigmas)
output_datum = ppxf_lib.ppxf_CaT(input_datum)
```

where `wavelengths`, `fluxes` and `sigmas` are 1d Numpy arrays contain the wavelengths, fluxes and (optionally) uncertainties of the spectra to be fit.


Christopher Usher

chris.usher@astro.se.su

2018-09-19  Liverpool
