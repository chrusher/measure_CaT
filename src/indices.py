#! /usr/bin/env python

import numpy as np

'''
Calculate spectral indices using the formula of Cenarro et al. (2001)
Includes index definitions for a range of commonly used indices.

Assumes constant linear dispersion
C01 = Cenarro et al 2001 MNRAS 326 959
'''

# convert vacuum wavelengths to air wavelengths
def vacuum_to_air(vacuum):
    air = vacuum / (1.0 + 2.735182e-4 + 131.4182 / vacuum**2 + 2.76249e8 / vacuum**4)
    return air

# calculates a sum of a band allowing for fractional pixels
def bandSum(band, wavelengths, fluxes, edges=True):

    dwavelength = wavelengths[1] - wavelengths[0]
    inband = (wavelengths > band[0] - dwavelength / 2) & (wavelengths < band[1] + dwavelength / 2)
    inwavelengths = wavelengths[inband]
    influxes = fluxes[inband]

    total = influxes[1:-1].sum()
    if edges:
        total += influxes[0] * (inwavelengths[:2].mean() - band[0]) / dwavelength
        total += influxes[-1] * (band[1] - inwavelengths[-2:].mean()) / dwavelength

    return total

'''
Calculate a spectral index using the formula provided in the appendices of
Cenarro et al. (2001)
'''
class Index:

    def __init__(self, name, mainpassbands, continuumpassbands, mainweights=None, atomic=True, flux=False, vacuum=False):

        self.name = name
        self.mainpassbands = np.array(mainpassbands)
        self.continuumpassbands = np.array(continuumpassbands)
        if vacuum:
            self.mainpassbands = vacuum_to_air(self.mainpassbands)
            self.continuumpassbands = vacuum_to_air(self.continuumpassbands)
            
        if mainweights is None:
            self.mainpassweights = np.ones(self.mainpassbands.shape[0])
        else:
            self.mainpassweights = np.array(mainweights)
        
            
        self.atomic = atomic
        #only relevent for SKiMS sky subtraction
        self.flux = flux

    def __str__(self):
        return self.name


    def __call__(self, wavelengths, fluxes, sigmas=None, normalized=False, calcerrors=True):
        return self.calculateIndex(wavelengths, fluxes, sigmas=sigmas, normalized=normalized, calcerrors=calcerrors)

    def calculateIndex(self, wavelengths, fluxes, sigmas=None, normalized=False, calcerrors=True):
        
        if sigmas is None:
            variances = fluxes
        else:
            variances = sigmas**2

        if fluxes.size != wavelengths.size or fluxes.size != variances.size:

            raise Exception('Input array size mismatch')    


        dwavelength = wavelengths[1] - wavelengths[0]

        #
        if not normalized:

            if wavelengths[0] > min(self.mainpassbands.min(), self.continuumpassbands.min()) or wavelengths[-1] < max(self.mainpassbands.max(), self.continuumpassbands.max()):
                raise Exception('Index not in wavelength range')
    
            #correspond to eqs a15 to a19 of C01
            Sigma1 = 0
            Sigma2 = 0
            Sigma3 = 0
            Sigma4 = 0
            Sigma5 = 0

            inversevariances =  1 / variances 
            sigma1seq = inversevariances
            sigma2seq = wavelengths * inversevariances
            sigma3seq = wavelengths**2 * inversevariances
            sigma4seq = fluxes * inversevariances
            sigma5seq = wavelengths * fluxes * inversevariances
            

            for passband in self.continuumpassbands:
                inpassband = (wavelengths > passband[0]) & (wavelengths < passband[1])
                Sigma1 += sigma1seq[inpassband].sum()
                Sigma2 += sigma2seq[inpassband].sum()
                Sigma3 += sigma3seq[inpassband].sum()
                Sigma4 += sigma4seq[inpassband].sum()
                Sigma5 += sigma5seq[inpassband].sum()

            # eq a14 of C01
            Lambda = Sigma1 * Sigma3 - Sigma2 **2

            #eqs a12 and a13 of C01
            alpha1 = (Sigma3 * Sigma4 - Sigma2 * Sigma5) / Lambda
            alpha2 = (Sigma1 * Sigma5 - Sigma2 * Sigma4) / Lambda

            continuum = alpha1 + alpha2 * wavelengths

            #Note that this has not been properly checked
            if calcerrors:
                
                error = 0
                fluxerror = 0
                continuumvariance = 0
                
                #eq a25
                continuummatrix = np.outer(np.ones(wavelengths.size), (sigma1seq * Sigma3 -  sigma2seq * Sigma2) / Lambda) + np.outer(wavelengths, (sigma2seq * Sigma1 - sigma1seq * Sigma2) / Lambda)
                
                #eq a26
                for passband in self.continuumpassbands:
                    inpassband = (wavelengths > passband[0]) & (wavelengths < passband[1])
                    continuumvariance += np.dot(continuummatrix[:,inpassband], variances[inpassband]) 
                
                #eqs a28 to a30
                a11 = (Sigma1 * Sigma3 * Sigma3 - Sigma2 * Sigma2 * Sigma3) / Lambda**2
                a12 = (Sigma2 * Sigma2 * Sigma2 - Sigma1 * Sigma2 * Sigma3) / Lambda**2
                a22 = (Sigma1 * Sigma1 * Sigma3 - Sigma1 * Sigma2 * Sigma2) / Lambda**2
                
            
                
                if self.flux:
                    #flux excess
                    for i in range(self.mainpassweights.shape[0]):
                        outerpassband = self.mainpassbands[i]
                        weight = self.mainpassweights[i]
                        inouterpassband = (wavelengths > outerpassband[0]) & (wavelengths < outerpassband[1])
                        
                        fluxerror += weight**2 * (variances[inouterpassband] + continuumvariance[inouterpassband]).sum()
                        error += weight**2 * ((continuum[inouterpassband]**2 * variances[inouterpassband] + fluxes[inouterpassband]**2 * continuumvariance[inouterpassband]) / continuum[inouterpassband]**4).sum()
                        
                        for j in inouterpassband.nonzero()[0]:
                            for k in range(self.mainpassbands.shape[0]):
                                innerpassband = self.mainpassbands[k]
                                ininnerpassband = (wavelengths > innerpassband[0]) & (wavelengths < innerpassband[1])
                                for l in ininnerpassband.nonzero()[0]:
                                    if (i != k) or (j != l):
                                        fluxerror += self.mainpassweights[i] * self.mainpassweights[k] * (a11 + a12 * (wavelengths[j] + wavelengths[l]) + a22 * wavelengths[j] * wavelengths[l])
                                        error += self.mainpassweights[i] * self.mainpassweights[k] * fluxes[j] * fluxes[l] * (a11 + a12 * (wavelengths[j] + wavelengths[l]) + a22 * wavelengths[j] * wavelengths[l]) / (continuum[j]**2 * continuum[l]**2 )
                    
                    fluxerror = fluxerror**.5 * dwavelength
                else:    
                    #eq a23
                    for i in range(self.mainpassweights.shape[0]):
                        
                        outerpassband = self.mainpassbands[i]
                        weight = self.mainpassweights[i]
                        inouterpassband = (wavelengths > outerpassband[0]) & (wavelengths < outerpassband[1])
                        
                        error += weight**2 * ((continuum[inouterpassband]**2 * variances[inouterpassband] + fluxes[inouterpassband]**2 * continuumvariance[inouterpassband]) / continuum[inouterpassband]**4).sum()
                        
                        for j in inouterpassband.nonzero()[0]:
                            for k in range(self.mainpassbands.shape[0]):
                                innerpassband = self.mainpassbands[k]
                                ininnerpassband = (wavelengths > innerpassband[0]) & (wavelengths < innerpassband[1])
                                for l in ininnerpassband.nonzero()[0]:
                                    if (i != k) or (j != l):
                                        error += self.mainpassweights[i] * self.mainpassweights[k] * fluxes[j] * fluxes[l] * (a11 + a12 * (wavelengths[j] + wavelengths[l]) + a22 * wavelengths[j] * wavelengths[l]) / (continuum[j]**2 * continuum[l]**2 )
    
                error = error**.5 * dwavelength                    

                
            else:
                error = 0
                fluxerror = 0

        # assume the spectra is normalized
        else:
            if wavelengths[0] > self.mainpassbands.min() or wavelengths[-1] < self.mainpassbands.max():
                raise Exception('Index not in wavelength range\nIndex {} {} Wavelengths {} {}'.format(self.mainpassbands.min(), self.mainpassbands.max(), wavelengths[0], wavelengths[-1]))
            continuum = np.zeros(fluxes.size) + 1
            error = 0

        

        total = 0

        normdepths = 1 - fluxes / continuum 

        # correspond to eq a9 of C01
        for i in range(self.mainpassbands.shape[0]):
            total += dwavelength * self.mainpassweights[i] * bandSum(self.mainpassbands[i], wavelengths, normdepths)
        
        if False:# self.name == 'All Metals':
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(wavelengths, fluxes)
            plt.plot(wavelengths, continuum)
            plt.vlines(self.continuumpassbands.flatten(), continuum.mean(), 1.1 * continuum.mean())
            plt.vlines(self.mainpassbands.flatten(), 0.9 * continuum.mean(), continuum.mean())
            plt.title(self.name)

        if not self.flux:
                        
            return total, error
        
            
        
        else:
            fluxtotal = 0
            excesses = fluxes - continuum
            for i in range(self.mainpassbands.shape[0]):
                fluxtotal += dwavelength * self.mainpassweights[i] * bandSum(self.mainpassbands[i], wavelengths, excesses)
                
            
            
            return total, error, fluxtotal, fluxerror 

'''
Calculate the flux ratio of two wavelength regions
'''
class Ratio:
    
    def __init__(self, name, blueband, redband, vacuum=False):
        
        self.name = name
        self.blueband = blueband
        self.redband = redband
        
        if vacuum:
            self.blueband = vacuum_to_air(self.blueband)
            self.redband = vacuum_to_air(self.redband)        
    
    def __call__(self, wavelengths, fluxes, sigmas=None, calcerrors=True):
        return self.calculateIndex(wavelengths, fluxes, sigmas=sigmas, calcerrors=calcerrors)

    def calculateIndex(self, wavelengths, fluxes, sigmas=None, calcerrors=True):
        
        if sigmas == None:
            sigmas = np.zeros(fluxes.size)
            
        
        if wavelengths[0] > self.blueband[0] or wavelengths[-1] < self.redband[-1]:
            raise Exception('Index not in wavelength range')


        varis = sigmas**2


        edges = (wavelengths[:-1] + wavelengths[1:]) / 2
        
        lower_edges = np.hstack((3 * wavelengths[0] / 2 - wavelengths[1] / 2, edges))
        upper_edges = np.hstack((edges, 3 * wavelengths[-1] / 2 - wavelengths[-2] / 2))
            
            
        blue_flux = 0
        blue_vari = 0
        
        red_flux = 0
        red_vari = 0    
        
        for i in range(wavelengths.size):
            
            
            if upper_edges[i] >= self.blueband[0] and lower_edges[i] <= self.blueband[1]:
                if lower_edges[i] <= self.blueband[0]:
                    
                    factor = (upper_edges[i] - self.blueband[0]) / (upper_edges[i] - lower_edges[i])
                    
                elif upper_edges[i] >= self.blueband[0]:
                    
                    factor = (self.blueband[1] - lower_edges[i]) / (upper_edges[i] - lower_edges[i])
                
                else:
                    factor = 1
                    
                blue_flux += fluxes[i] * factor
                blue_vari += varis[i] * factor**2
                
            if upper_edges[i] >= self.redband[0] and lower_edges[i] <= self.redband[1]:
                
                if lower_edges[i] <= self.redband[0]:
                    
                    factor = (upper_edges[i] - self.redband[0]) / (upper_edges[i] - lower_edges[i])
                    
                elif upper_edges[i] >= self.redband[0]:
                    
                    factor = (self.redband[1] - lower_edges[i]) / (upper_edges[i] - lower_edges[i])
                
                else:
                    factor = 1
                    
                red_flux += fluxes[i] * factor
                red_vari += varis[i] * factor**2            
        

#        print blue_flux, blue_vari**0.5
#        print red_flux, red_vari**0.5

        ratio = blue_flux / red_flux
        
        if calcerrors:
            error = (blue_vari / red_flux**2 + blue_flux**2 / red_flux**4 * red_vari)**0.5
        else:
            error = 0.
        
        return ratio, error
    

'''
Only meant for normalized spectra
Based on Armandroff & Zinn (1988), used in Foster et al. (2010, 2011),
Usher et al. (2012, 2015, 2018)
'''
CaT_gc = Index('CaT gc', np.array([[8490.0, 8506.0], [8532.0, 8552.0], [8653.0, 8671.0]]), None, np.array([1, 1, 1]))

'''
Armandroff & Zinn (1988) definition
'''
CaT_AZ_1 = Index('CaT A&Z 1', np.array([[8490.0, 8506.0]]), np.array([[8474.0, 8489.0], [8521.0, 8531.0]]), np.array([1]))
CaT_AZ_2 = Index('CaT A&Z 2', np.array([[8532.0, 8552.0]]), np.array([[8521.0, 8531.0], [8555.0, 8595.0]]), np.array([1]))
CaT_AZ_3 = Index('CaT A&Z 3', np.array([[8653.0, 8671.0]]), np.array([[8626.0, 8650.0], [8695.0, 8725.0]]), np.array([1]))

def CaT_AZ(wavelengths, fluxes, sigmas=None, normalized=False, calcerrors=True):
    az1, az1_e = CaT_AZ_1(wavelengths, fluxes, sigmas, normalized, calcerrors) 
    az2, az2_e = CaT_AZ_2(wavelengths, fluxes, sigmas, normalized, calcerrors)
    az3, az3_e = CaT_AZ_3(wavelengths, fluxes, sigmas, normalized, calcerrors)
    az = az1 + az2 + az3
    aze = (az1_e**2 + az2_e**2 + az3_e**2)**0.5
    return az, aze

'''
Cenarro et al. (2001) definitions
'''
CaT_C01 = Index('CaT C01', np.array([[8484.0, 8513.0], [8522.0, 8562.0], [8642.0, 8682.0]]), np.array([[8474.0, 8484.0], [8563.0, 8577.0], [8619.0, 8642.0], [8700.0, 8725.0], [8776.0, 8792.0]]), np.array([1, 1, 1]))
PaT_C01 = Index('PaT C01', np.array([[8461.0, 8474.0], [8577.0, 8619.0], [8730.0, 8772.0]]), np.array([[8474.0, 8484.0], [8563.0, 8577.0], [8619.0, 8642.0], [8700.0, 8725.0], [8776.0, 8792.0]]), np.array([1, 1, 1]))
CaTS_C01 = Index('CaT* C01', np.array([[8484.0, 8513.0], [8522.0, 8562.0], [8642.0, 8682.0], [8461.0, 8474.0], [8577.0, 8619.0], [8730.0, 8772.0]]), np.array([[8474.0, 8484.0], [8563.0, 8577.0], [8619.0, 8642.0], [8700.0, 8725.0], [8776.0, 8792.0]]), np.array([1, 1, 1, -.93, -.93, -.93]))


'''
Fe86 weak metal line index from Usher et al. (2015)
'''
Fe86 = Index('All Metals', np.array([[8375.5, 8392.0],
                                        [8410.4, 8414.0],
                                        [8424.5, 8428.0],
                                        [8432.5, 8440.9],
                                        [8463.7, 8473.0],
                                        [8512.8, 8519.0],
                                        [8580.8, 8583.5],
                                        [8595.7, 8601.0],
                                        [8609.0, 8613.5],
                                        [8620.2, 8623.3],
                                        [8673.2, 8676.5],
                                        [8686.8, 8690.7],
                                        [8820.5, 8827.0],
                                        [8836.0, 8840.5]]), 
                np.array([[8392.0, 8393.5],
                        [8399.4, 8400.9],
                        [8402.7, 8410.3],
                        [8414.5, 8422.1],
                        [8428.6, 8432.3],
                        [8441.4, 8445.2],
                        [8447.9, 8449.4],
                        [8451.5, 8455.4],
                        [8458.0, 8463.0],
                        [8474.0, 8493.3],
                        [8505.3, 8512.1],
                        [8519.2, 8525.2],
                        [8528.3, 8531.3],
                        [8552.3, 8554.9],
                        [8557.5, 8580.4],
                        [8583.9, 8595.3],
                        [8601.2, 8608.4],
                        [8613.9, 8619.4],
                        [8624.3, 8646.6],
                        [8649.8, 8652.5],
                        [8676.9, 8678.1],
                        [8684.0, 8686.1],
                        [8692.7, 8697.6],
                        [8700.3, 8708.9],
                        [8714.5, 8726.8],
                        [8731.5, 8733.2],
                        [8737.6, 8740.8],
                        [8743.3, 8746.1],
                        [8754.5, 8755.4],
                        [8759.0, 8762.2],
                        [8768.0, 8771.5],
                        [8775.5, 8788.7],
                        [8797.6, 8802.2],
                        [8811.0, 8820.0],
                        [8828.0, 8835.0]]))




'''
Lick indices
definitions from http://astro.wsu.edu/worthey/html/index.table.html
'''
Hdelta_A = Index('Hdelta_A', np.array([[4083.500, 4122.250]]), np.array([[4041.600, 4079.750], [4128.500, 4161.000]])) 
Hdelta_F = Index('Hdelta_F', np.array([[4091.000, 4112.250]]), np.array([[4057.250, 4088.500], [4114.750, 4137.250]])) 
CN_1 = Index('CN_1', np.array([[4142.125, 4177.125]]), np.array([[4080.125, 4117.625], [4244.125, 4284.125]]), atomic=False) 
CN_2 = Index('CN_2', np.array([[4142.125, 4177.125]]), np.array([[4083.875, 4096.375], [4244.125, 4284.125]]), atomic=False)  
Ca4227 = Index('Ca4227', np.array([[4222.250, 4234.750]]), np.array([[4211.000, 4219.750], [4241.000, 4251.000]]))
Hgamma_A = Index('Hgamma_A', np.array([[4319.750, 4363.500]]), np.array([[4283.500, 4319.750], [4367.250, 4419.750]]))
Hgamma_F = Index('Hgamma_F', np.array([[4331.250, 4352.250]]), np.array([[4283.500, 4319.750], [4354.750, 4384.750 ]]))
G4300 = Index('G4300', np.array([[4281.375, 4316.375]]), np.array([[4266.375, 4282.625], [4318.875, 4335.125]]))
Fe4383 = Index('Fe4383', np.array([[4369.125, 4420.375]]), np.array([[4359.125, 4370.375], [4442.875, 4455.375]]))
Ca4455 = Index('Ca4455', np.array([[4452.125, 4474.625]]), np.array([[4445.875, 4454.625], [4477.125, 4492.125]])) 
Fe4531 = Index('Fe4531', np.array([[4514.250, 4559.250]]), np.array([[4504.250, 4514.250], [4560.500, 4579.250]]))
Fe4668 = Index('Fe4668', np.array([[4634.000, 4720.250]]), np.array([[4611.500, 4630.250], [4742.750, 4756.500]])) 
H_beta = Index('H_beta', np.array([[4847.875, 4876.625]]), np.array([[4827.875, 4847.875], [4876.625, 4891.625]])) 
Fe5015 = Index('Fe5015', np.array([[4977.750, 5054.000]]), np.array([[4946.500, 4977.750], [5054.000, 5065.250]]))
Mg_1 = Index('Mg_1', np.array([[5069.125, 5134.125]]), np.array([[4895.125, 4957.625], [5301.125, 5366.125]]), atomic=False)
Mg_2 = Index('Mg_2', np.array([[5154.125, 5196.625]]), np.array([[4895.125, 4957.625], [5301.125, 5366.125]]), atomic=False)
Mg_b = Index('Mg_b', np.array([[5160.125, 5192.625]]), np.array([[5142.625, 5161.375], [5191.375, 5206.375]]))
Fe5270 = Index('Fe5270', np.array([[5245.650, 5285.650]]), np.array([[5233.150, 5248.150], [5285.650, 5318.150]]))
Fe5335 = Index('Fe5335', np.array([[5312.125, 5352.125]]), np.array([[5304.625, 5315.875], [5353.375, 5363.375]]))
Fe5406 = Index('Fe5406', np.array([[5387.500, 5415.000]]), np.array([[5376.250, 5387.500], [5415.000, 5425.000]])) 
Fe5709 = Index('Fe5709', np.array([[5696.625, 5720.375]]), np.array([[5672.875, 5696.625], [5722.875, 5736.625]])) 
Fe5782 = Index('Fe5782', np.array([[5776.625, 5796.625]]), np.array([[5765.375, 5775.375], [5797.875, 5811.625]])) 
Na_D = Index('Na_D', np.array([[5876.875, 5909.375]]), np.array([[5860.625, 5875.625], [5922.125, 5948.125]])) 
TiO_1 = Index('TiO_1', np.array([[5936.625, 5994.125]]), np.array([[5816.625, 5849.125], [6038.625, 6103.625]]), atomic=False)
TiO_2 = Index('TiO_2', np.array([[6189.625, 6272.125]]), np.array([[6066.625, 6141.625], [6372.625, 6415.125]]), atomic=False) 

def Fe_mean(wavelengths, fluxes, sigmas=None, normalized=False, calcerrors=True):
    fe52, fe52_e = Fe5270(wavelengths, fluxes, sigmas, normalized, calcerrors) 
    fe53, fe53_e = Fe5335(wavelengths, fluxes, sigmas, normalized, calcerrors)
    fe_mean = (fe52 + fe53) / 2
    fe_mean_e = (fe52_e**2 + fe53_e**2)**0.5 / 2
    return fe_mean, fe_mean_e

Fe_mean.name = 'Fe_mean'




#Conroy & van Dokkum (2012) definitions
TiO89 = Ratio('TiO89', np.array([8835.0, 8855.0]), np.array([8870.0, 8890.0]), vacuum=True)
Na59 = Index('Na59', np.array([[5878.5, 5911.0]]), np.array([[5862.2, 5877.2], [5923.7, 5949.7]]), vacuum=True)
Na82 = Index('Na82', np.array([[8177.0, 8205.0]]), np.array([[8170.0, 8177.0], [8205.0, 8215.0]]), vacuum=True)
Ca39 = Index('Ca39', np.array([[3899.5, 4003.5]]), np.array([[3806.5, 3833.8], [4020.7, 4052.4]]), vacuum=True)
MgI52b = Index('MgI52b', np.array([[5165.0, 5220.0]]), np.array([[5125.0, 5165.0], [5220.0, 5260.0]]), vacuum=True)
Mg88 = Index('Mg88', np.array([[8801.9, 8816.9]]), np.array([[8777.4, 8789.4], [8847.4, 8857.4]]), vacuum=True)

#Vazdekis et al. (2012)
Na_D_v = Index('Na_D', np.array([[8180.0, 8200.0]]), np.array([[8164.0, 8173.0], [8233.0, 8244.0]])) # 2012MNRAS.424..157V

#Cennaro et al. (2009)
MgI = Index('MgI', np.array([[8802.5, 8811.0]]), np.array([[8781.0, 8787.0], [8831.0, 8835.5]]), np.array([1]))


#Nelan et al. (2005)
HaA = Index('HaA', np.array([[6554.0, 6575.0]]), np.array([[6515.0, 6540.0], [6575.0, 6585.0]]), np.array([1]))
HaF = Index('HaF', np.array([[6554.0, 6568.0]]), np.array([[6515.0, 6540.0], [6568.0, 6575.0]]), np.array([1]))


#Pastorello et al. (2014)    
CaT_gal = Index('CaT gal', np.array([[8483.0, 8513.0], [8527.0, 8557.0], [8647.0, 8677.0]]), np.array([[8474.0, 8483.0], [8514.0, 8526.0], [8563.0, 8577.0], [8619.0, 8642.0], [8680.0, 8705.0]]), np.array([0.4, 1.0, 1.0]))
