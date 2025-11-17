import argparse
import numpy as np
import h5py
from pathlib import Path
from ..physics.constants import c, h, k_B, H_alpha, H_beta, H_gamma, H_delta, Ca_H, Ca_K, Mg_b

def lambda_0_dom(z: float, lambda_instrument_data: np.ndarray) -> np.ndarray:
    '''
    This function creates the wavelength domain without redshift for the spectrum, that is,
    the one that would be measured if the radiation source were not moving relative to us.
    ---
    Arguments:
    z: Redshift of the spectrum to generate.
    lambda_instrument_data: Domain of wavelength values measured by the same instrument.
    '''
    return lambda_instrument_data/(1+z)

def blackbody(lambda_0_data: np.ndarray, T: float) -> np.ndarray:
    '''
    This function returns the radiance data of a black body at temperature T for the
    considered wavelengths.
    ---
    Arguments:
    lambda0_data: Domain of wavelength values emitted by the source.
    T: Temperature of the black body.
    '''
    return ((2*h*c**2)/(lambda_0_data**5)) / (np.exp((h*c)/(lambda_0_data*k_B*T))-1)

def add_lines(lambda_0_data: np.ndarray, base_spectrum: np.ndarray, random_parameters: bool = False, depth = 0.5, gamma = 5e-10, depth_metals = 0.2, gamma_metals = 2e-10) -> np.ndarray:
    '''
    This function adds the main hydrogen absorption lines to a black-body spectrum.
    ---
    Arguments:
    lambda_0_data: Domain of wavelength values emitted by the source.
    base_spectrum: Bse spectrum to add the lines
    depth: Depth of the absortion line (values between 0 and 1)
    gamma: Thickness of the absortion line
    depth_metals: Depth of the absortion line for metals (values between 0 and 1)
    gamma_metals: Thickness of the absortion line for metals
    '''
    if random_parameters:
        depth = np.random.uniform(0.5, 0.9)
        gamma = np.random.uniform(5e-10, 2e-9)
        depth_metals = np.random.uniform(0.2, 0.7)
        gamma_metals = np.random.uniform(2e-10, 1e-9)

    H_alpha_lines = depth * gamma**2 / ((lambda_0_data - H_alpha)**2 + gamma**2)
    H_beta_lines = depth * gamma**2 / ((lambda_0_data - H_beta)**2 + gamma**2)
    H_gamma_lines = depth * gamma**2 / ((lambda_0_data - H_gamma)**2 + gamma**2)
    H_delta_lines = depth * gamma**2 / ((lambda_0_data - H_delta)**2 + gamma**2)
    Ca_H_lines = depth_metals * gamma_metals**2 / ((lambda_0_data - Ca_H)**2 + gamma_metals**2)
    Ca_K_lines = depth_metals * gamma_metals**2 / ((lambda_0_data - Ca_K)**2 + gamma_metals**2)
    Mg_b_lines = depth_metals * gamma_metals**2 / ((lambda_0_data - Mg_b)**2 + gamma_metals**2)
    
    lorentzian_profile = H_alpha_lines + H_beta_lines + H_gamma_lines + H_delta_lines + Ca_H_lines + Ca_K_lines + Mg_b_lines
    
    return base_spectrum * (1 - lorentzian_profile)

def noise_generator(emission_data: np.ndarray, random_parameters:bool = False, standar_dev: float = 1e12) -> np.ndarray:
    '''
    This function adds Gaussian noise to a spectrum.
    ---
    Arguments:
    emission_data: Radiance data emitted by the source (black body + lines).
    '''
    if random_parameters:
        standar_dev = np.random.uniform(1e12, 1e13)       
    
    return emission_data + np.random.normal(0, standar_dev, len(emission_data))

def spectrum_generator(T, z, lambda_instrument_data, random_parameters: bool = False, scale_factor: float = 0.1, noise_standar_dev:float = 1e12, line_params = None):
    '''
    This function generates a synthetic spectrum of a source at temperature T and with redshift
    z. The spectrum corresponds to that of a black body to which the first four
    hydrogen absorption lines and Gaussian noise are added. This spectrum is limited to the
    wavelength domain that a single instrument can measure.
    ---
    Arguments:
    T: Temperature of the source.
    z: Redshift of the source.
    lambda_instrument_data: Wavelength value domain in which a single instrument measures.
    scale_factor: Scale factor of the spectrum
    '''
        

    lambda_0_data = lambda_0_dom(z, lambda_instrument_data)
    if random_parameters:
        scale_factor = np.random.uniform(0.1, 5.0)

    if line_params is not None:
        emission_data = add_lines(lambda_0_data, blackbody(lambda_0_data, T), random_parameters, *line_params) * scale_factor
    else:
        emission_data = add_lines(lambda_0_data, blackbody(lambda_0_data, T), random_parameters) * scale_factor
    spectrum = noise_generator(emission_data, random_parameters, noise_standar_dev)
    return z, spectrum

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Blackbody radiation with absortion lines data generator")
    parser.add_argument("--min_lambda", type=float, default=400, help="Minimum wave lenght measured by the instrument in nm. Default 400")
    parser.add_argument("--max_lambda", type=float, default=900, help="Max wave lenght measured by the instrument in nm. Default 900")
    parser.add_argument("-n_lambda", type=float, default=1000, help="Amount of channels of the instrument. Default 1000")
    parser.add_argument("-n_spectra", type=float, default=5000, help="Amount of spectrums generated. Default 5000")
    parser.add_argument("--random_parameters", action="store_true" , help="Sets certain variables in a random value in a range: Gaussian noise std (1e12 to 1e13), depth (0.5 to 0.9), gamma (5e-10 to 2e-9), depth_metals (0.2 to 0.7) and gamma_metals (2e-10 to 1e-9)")
    parser.add_argument("--noise_std", type=float, default=5e12, help="Gausina noise standard deviation")
    parser.add_argument("--depth", type=float, default=0.5, help="Non metalic elements absortion line depth between 0 and 1")
    parser.add_argument("--depth_metals", type=float, default=0.2, help="Metalic elements absortion line depth between 0 and 1")
    parser.add_argument("--gamma", type=float, default= 5e-10, help="Non metalic elements absortion line width in nm")
    parser.add_argument("--gamma_metals", type=float, default=2e-10, help="Metalic elements absortion line width in nm")
    
    args = parser.parse_args()
    
    # Wave lenght domain
    lambda_min_measured = args.min_lambda
    lambda_max_measured = args.max_lambda
    N_lambda = args.n_lambda
    lambda_instrument_data = np.linspace(lambda_min_measured, lambda_max_measured, N_lambda) * 1e-9

    standar_dev = args.noise_std
    depth = args.depth
    depth_metals = args.depth_metals
    gamma = args.gamma
    gamma_metals = args.gamma_metals
    line_params = [depth, gamma, depth_metals, gamma_metals]

    N_spectra = args.n_spectra

    np.random.seed(42)
    T_values = np.random.uniform(3000, 15000, N_spectra)
    z_values = np.random.uniform(0, 1, N_spectra)
    
    spectra_data = np.zeros((N_spectra+1, N_lambda+1))
    spectra_data[0, 1:] = lambda_instrument_data

    for i in range(N_spectra):
        spectra_data[i+1,0], spectra_data[i+1,1:] = spectrum_generator(T_values[i], z_values[i], lambda_instrument_data, args.random_parameters, noise_standar_dev=standar_dev, line_params=line_params)
    
    FILE_DIR = Path(__file__).resolve().parent
    h5_path = FILE_DIR.parent.parent / "data" / "SpectraData.h5"

    with h5py.File(h5_path, 'w') as f:
        # Essential metadata
        f.attrs['title'] = "Synthetic Spectra Database"
        f.attrs['description'] = (
            f'This file contains {N_spectra} synthetic spectra for wavelengths between {lambda_min_measured} and {lambda_max_measured}, with redshift values z between 0 and 1 and temperatures between 3000 and 15000 K.'
        )

        # Save the common domain (only once)
        wavelength_ds = f.create_dataset('Wavelength Domain', data=spectra_data[0, 1:])
        wavelength_ds.attrs['description'] = (
            f'Wavelength domain shared by all spectra: {N_lambda} values between {lambda_min_measured} and {lambda_max_measured}.'
        )
        wavelength_ds.attrs['units'] = 'm'

        redshift_ds = f.create_dataset('Redshift values', data=z_values)
        redshift_ds.attrs['description'] = (f'{N_lambda} redshift values between 0 and 1, each corresponding to a different spectrum.')

        temperature_ds = f.create_dataset('Temperature values', data=T_values)
        temperature_ds.attrs['description'] = (f'{N_lambda} temperature values between 3000 K and 15000 K, each corresponding to a different spectrum.')
        temperature_ds.attrs['units'] = 'K'

        spectra_grp = f.create_group('Spectra')
        for i in range(N_spectra):
            # Create dataset for each spectrum
            spec_ds = spectra_grp.create_dataset(
                f'spec_{i}',
                data=spectra_data[i+1, 1:]  # shape (1000,)
            )
            # METADATA PER SPECTRUM
            spec_ds.attrs['description'] = f'Spectrum number {i}'
            spec_ds.attrs['z'] = z_values[i]  # Corresponding redshift
            spec_ds.attrs['T'] = T_values[i]  # Corresponding temperature
            spec_ds.attrs['units'] = 'W m^-3'
