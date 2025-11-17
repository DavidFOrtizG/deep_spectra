import numpy as np
from ..physics.constants import c, h, k_B

def Lambda0_Dom(z: float, lambda_instrument_data: np.ndarray):
    '''
    This function creates the wavelength domain without redshift for the spectrum, that is,
    the one that would be measured if the radiation source were not moving relative to us.
    ---
    Arguments:
    z: Redshift of the spectrum to generate.
    lambda_instrument_data: Domain of wavelength values measured by the same instrument.
    '''
    return lambda_instrument_data/(1+z)

def blackbody(lambda0_data, T):
    '''
    This function returns the radiance data of a black body at temperature T for the
    considered wavelengths.
    ---
    Arguments:
    lambda0_data: Domain of wavelength values emitted by the source.
    T: Temperature of the black body.
    '''
    return ((2*h*c**2)/(lambda0_data**5)) / (np.exp((h*c)/(lambda0_data*k_B*T))-1)

if __name__ == "__main__":
    print(h, c, k_B)