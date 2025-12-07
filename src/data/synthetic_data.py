import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from src.physics.constants import c, h, k_B, H_alpha, H_beta, H_gamma, H_delta, Ca_H, Ca_K, Mg_b
import yaml

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

def load_config(config_path='params.yaml'):
    """Loads configuration parameters from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}. Please check the path.")
        exit(1)

# --- Main execution block updated to use YAML ---
if __name__ == "__main__":
    
    # 1. Use argparse ONLY to get the configuration file path (optional, default is params.yaml)
    parser = argparse.ArgumentParser(description="Blackbody radiation with absortion lines data generator")
    parser.add_argument('--config', type=str, default='params.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    
    # 2. Load the full configuration
    full_config = load_config(args.config)
    
    # 3. Extract the 'synthetic_data' block
    try:
        synthetic_config = full_config['synthetic_data']
    except KeyError:
        print("Error: 'synthetic_data' key not found in the YAML file. Check your structure.")
        exit(1)
        
    # --- Assigning parameters based on YAML structure ---
    
    # Access Instrument Parameters
    instrument_params = synthetic_config['instrument']
    lambda_min_measured = instrument_params['min_lambda']
    lambda_max_measured = instrument_params['max_lambda']
    N_lambda = instrument_params['n_lambda']
    
    # Wave length domain calculation (identical to original code)
    # Note: Added int() casting to N_lambda as np.linspace usually expects an integer count.
    lambda_instrument_data = np.linspace(lambda_min_measured, lambda_max_measured, int(N_lambda)) * 1e-9

    # Access Dataset Parameters
    dataset_params = synthetic_config['dataset']
    N_spectra = dataset_params['n_spectra']
    random_parameters_enabled = dataset_params['random_parameters']

    # Access Model Parameters
    model_params = synthetic_config['model_params']
    
    # Noise
    standar_dev = model_params['noise_std']
    
    # Non-metals
    non_metals_params = model_params['non_metals']
    depth = non_metals_params['depth']
    gamma = non_metals_params['gamma']

    
    # Metals
    metals_params = model_params['metals']
    depth_metals = metals_params['depth']
    gamma_metals = metals_params['gamma']
    
    # Final array (identical to original code)
    line_params = [depth, gamma, depth_metals, gamma_metals]

    print(f"Configuration successfully loaded from {args.config}.")
    print(f"Generating {N_spectra} spectra over {N_lambda} channels.")
    print(f"Random parameters enabled: {random_parameters_enabled}")

    np.random.seed(42)
    T_values = np.random.uniform(3000, 15000, N_spectra)
    z_values = np.random.uniform(0, 1, N_spectra)
    
    spectra_data = np.zeros((N_spectra+1, N_lambda+1))
    spectra_data[0, 1:] = lambda_instrument_data

    for i in tqdm(range(N_spectra), total=N_spectra, desc="Generating spectra"):
        spectra_data[i+1,0], spectra_data[i+1,1:] = spectrum_generator(T_values[i], z_values[i], lambda_instrument_data, random_parameters_enabled, noise_standar_dev=standar_dev, line_params=line_params)
    
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
        for i in tqdm(range(N_spectra), total=N_spectra, desc=f"Saving data to {h5_path}"):
            # Create dataset for each spectrum
            spec_ds = spectra_grp.create_dataset(
                f'spec_{i}',
                data=spectra_data[i+1, 1:]  
            )
            # METADATA PER SPECTRUM
            spec_ds.attrs['description'] = f'Spectrum number {i}'
            spec_ds.attrs['z'] = z_values[i]  # Corresponding redshift
            spec_ds.attrs['T'] = T_values[i]  # Corresponding temperature
            spec_ds.attrs['units'] = 'W m^-3'
