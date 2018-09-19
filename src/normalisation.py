#! /usr/bin/env python

'''
Continuum normalise spectra
'''

from __future__ import print_function
import numpy as np


def normalise(wavelengths, fluxes, order, mask=None, iters=None, lowsigma=0.005, upsigma=0.010, ivars=None, plot=False):
    return fluxes / getnormalisation(wavelengths, fluxes, order, mask, iters, lowsigma, upsigma, ivars, plot)

def getnormalisation(wavelengths, fluxes, order, maskregions=None, iters=None, lowsigma=0.005, upsigma=0.010, ivars=None, plot=False):

    if wavelengths.size != fluxes.size:
        raise Exception('Wavelength and flux arrays are different size')
    
    if ivars is None or ivars.size != wavelengths.size:
        ivars = np.ones(wavelengths.size)
    
    regionmask = np.ones(wavelengths.size, dtype=np.bool_)
    if maskregions is not None:
        for maskregion in maskregions:
            regionmask = regionmask & ~((wavelengths > maskregion[0]) & (wavelengths < maskregion[1]))

    polydomain = np.linspace(-1, 1, wavelengths.size) #Domain of the fitted polynomial

    polys = [np.poly1d([1]), np.poly1d([1, 0])]

    while len(polys) <= order:

        newpoly = np.poly1d([2, 0]) * polys[-1] - polys[-2]
        polys.append(newpoly)

    polys = polys[-1::-1]

    basisvectors = np.zeros((wavelengths.size, order + 1))
    weightedbasisvectors = basisvectors.copy()
    
    for i in range(basisvectors.shape[1]):
        basisvectors[:, i] = polys[i](polydomain)
        weightedbasisvectors[:, i] = polys[i](polydomain) * ivars


    pixelmask = regionmask
    masksize = fluxes[pixelmask].size

    beta = np.linalg.lstsq(weightedbasisvectors[pixelmask,:], fluxes[pixelmask] * ivars[pixelmask])[0]
    model = np.dot(basisvectors, beta)
    
    if plot:
        print(masksize)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('initial')
        plt.plot(wavelengths, fluxes, 'k')
        plt.plot(wavelengths[~pixelmask], fluxes[~pixelmask], 'r.')
        plt.plot(wavelengths, model, 'b')    
    
    count = 1
    if iters:
        while True:

            oneprecut = ((fluxes - model) / model > -lowsigma) & ((fluxes - model) / model < upsigma)

            pixelmask = oneprecut#& regionmask

            maskedfluxes = fluxes[pixelmask] 
            beta = np.linalg.lstsq(weightedbasisvectors[pixelmask,:], maskedfluxes * ivars[pixelmask])[0]
            model = np.dot(basisvectors, beta)
            
            if plot:
                print(count, maskedfluxes.size)

                import matplotlib.pyplot as plt
                plt.figure()
                plt.title('iter ' + str(count))
                plt.plot(wavelengths, fluxes, 'k')
                plt.plot(wavelengths[~pixelmask], fluxes[~pixelmask], 'r.')
                plt.plot(wavelengths, model, 'b')
            
            count += 1
            
            if masksize == maskedfluxes.size:
                #print masksize
                break
            else:
                masksize = maskedfluxes.size
                #print masksize
    

    return model


#mask in Usher et al. (2018)
atlas_mask = np.array([
[8424., 8427.7554],
[8433.61475, 8436.7358],
[8438.2969, 8441.3774],
[8445.6411, 8448.00635],
[8449.3579, 8451.2168],
[8455.3589, 8460.1372],
[8463.0996, 8469.5337],
[8470.93165, 8472.626],
[8479.7456, 8482.9683],
[8487.3379, 8488.6968],
[8495.1504, 8505.478],
[8513.09375, 8515.7759],
[8517.6424, 8518.8493],
[8525.1914, 8528.26075],
[8531.1111, 8532.3625],
[8534.4458, 8551.5747],
[8554.6543, 8557.30565],
[8574.0947, 8576.10985],
[8581.1714, 8583.6172],
[8595.6426, 8600.54395],
[8610.0527, 8612.8086],
[8620.7783, 8622.8906],
[8641.49315, 8642.78955],
[8647.84715, 8649.707],
[8653.5996, 8655.0713],
[8656.58595, 8669.06055],
[8673.43895, 8676.21535],
[8678.60155, 8680.6999],
[8681.8999, 8684.0708],
[8686.89355, 8690.3257],
[8691.7032, 8692.9462],
[8698.10695, 8700.1509],
[8702.6117, 8703.86925],
[8709.33445, 8710.90975],
[8712.11465, 8714.082],
[8717.2254, 8718.4661],
[8727.73095, 8729.86915],
[8734.32275, 8736.76855],
[8740.5259, 8742.7993],
[8746.7718, 8752.64015],
[8756.229, 8758.0244],
[8763.2373, 8764.76445],
[8766.0036, 8767.5762],
[8772.39895, 8774.59325],
[8778.1309, 8779.34855],
[8784.0626, 8785.3163],
[8789.6538, 8791.06105],
[8792.4673, 8794.18215],
[8795.88485, 8797.14055],
[8803.729, 8809.36525],
[8818.8768, 8820.08215],
[8823.03025, 8825.32425],
[8832.83215, 8834.06895],
[8837.4673, 8839.82375],
[8846.05515, 8847.3599],
])

#mask in Usher et al. (2010)
newmask = np.array([[8432.0, 8438.0],
[8462.5, 8474.0],
[8491.0, 8506.4],
[8510.0, 8520.0],
[8524.0, 8552.5],
[8553.8, 8558.0],
[8579.7, 8585.5],
[8594.7, 8600.9],
[8609.3, 8615.0],
[8619.6, 8624.0],
[8646.5, 8677.0],
[8685.5, 8693.6],
[8708.2, 8715.0],
[8733.0, 8738.0],
[8740.0, 8743.5],
[8745.6, 8753.5],
[8755.6, 8758.7],
[8762.2, 8768.0],
[8769.9, 8775.8],
[8777.5, 8781.0],
[8783.3, 8785.8],
[8788.1, 8795.3],
[8800.0, 8810.5],
[8817.7, 8828.0],
[8832.3, 8842.2]])

