"""
Galaxy Image Simulator using Galsim
Simulates HSC and HST observations of galaxies from the COSMOSWeb catalog
"""

import numpy as np
import galsim
from galgenai.cosmos.cosmos_catalog import COSMOSWebCatalog
import matplotlib.pyplot as plt
from surveycodex import get_survey
from surveycodex.utilities import mag2counts, mean_sky_level

def generate_image_from_row(sim: GalaxySim, galaxy_row, filter_names=None):
    """
    Generate multi-band images for a single galaxy

    Parameters:
    -----------
    sim : GalaxySim
        Galaxy simulator instance
    galaxy_row : astropy.table.Row
        Single row from catalog
    filter_names : list
        List of filter names to simulate

    Returns:
    --------
    dict : Dictionary with filter names as keys and image arrays as values
    """
    if filter_names is None:
        filter_names=sim.survey.available_filters
    galaxy_params_multi_band = {}
    psf_params_multi_band = {}

    for filter_name in filter_names:
        filter_obj = sim.survey.get_filter(filter_name)

        galaxy_params_multi_band[filter_name] = {
            "magnitude": float(galaxy_row[f"mag_model_hsc-{filter_name}"]),
            "hlr": float(galaxy_row['radius_sersic'] * 3600),
            "sersic_n": float(galaxy_row['sersic']),
            "axis_ratio": float(galaxy_row['axratio_sersic']),
            "position_angle": float(galaxy_row['angle_sersic']),
        }

        psf_params_multi_band[filter_name] = {
            "fwhm": filter_obj.psf_fwhm.value,
            "beta": 3.0
        }

    # Generate images
    multi_band_images, multi_band_pixel_variance = sim.simulate_galaxy(
        galaxy_params_multiband=galaxy_params_multi_band,
        add_noise="all",
        psf_type="moffat",
        psf_params_multiband=psf_params_multi_band,
    )

    # Convert to numpy arrays (copy so GalSim Image objects can be freed immediately)
    images_dict = {band: img.array.copy() for band, img in multi_band_images.items()}
    pixel_variance_dict = {band: pv.copy() for band, pv in multi_band_pixel_variance.items()}
    return images_dict, pixel_variance_dict, galaxy_params_multi_band

def sim_single_band_sersic_galaxy(flux, hlr, sersic_n, axis_ratio, position_angle, gsparams=None):
        """
        Create galaxy light profile from catalog parameters

        Parameters:
        -----------
        flux : float
            Flux in photons
        hlr : float
            Half-light radius in arcsec
        sersic_n : float
            Sersic index
        axis_ratio : float
            b/a ratio (default: 1.0 for circular)
        position_angle : float
            PA in degrees
        gsparams : galsim.GSParams, optional
            GSParams object for FFT settings

        Returns:
        --------
        galsim.GSObject
            Galaxy profile
        """
        # Create Sersic profile with GSParams
        gal = galsim.Sersic(n=sersic_n, half_light_radius=hlr, flux=flux, gsparams=gsparams)

        # Apply shear for ellipticity
        if axis_ratio < 1.0:
            g = (1 - axis_ratio) / (1 + axis_ratio)  # reduced shear
            gal = gal.shear(g=g, beta=position_angle * galsim.degrees)

        return gal


class GalaxySim:
    """Galaxy image simulator using Galsim"""

    def __init__(self, catalog=None, survey_name='HSC', image_size=53, random_seed=None, max_fft_size=512):
        """
        Initialize the simulator

        Parameters
        ----------
        catalog : COSMOSWebCatalog, optional
            Galaxy catalog to use (default: None)
        survey_name : str, optional
            Survey name for pixel scale and filter definitions (default: 'HSC')
        image_size : int, optional
            Size of simulated images in pixels (default: 53)
        random_seed : int, optional
            Random seed for reproducibility. If None, uses 12345 (default: None)
        max_fft_size : int, optional
            Maximum FFT size for Galsim operations (default: 512)
        """
        self.catalog = catalog
        self.survey = get_survey(survey_name=survey_name)
        self.survey_name = survey_name
        self.image_size = image_size
        self.random_seed = random_seed
        self.max_fft_size = max_fft_size
        self.rng = galsim.BaseDeviate(random_seed or 12345)
        self.gsparams = galsim.GSParams(maximum_fft_size=max_fft_size)

    def get_psf(self, psf_type, psf_params, gsparams=None):
        """
        Create PSF for the instrument

        Parameters:
        -----------
        psf_type : str
            Type of PSF ('moffat' or 'gaussian')
        psf_params : dict
            PSF parameters (fwhm required, beta for Moffat)
        gsparams : galsim.GSParams, optional
            GSParams object for FFT settings

        Returns:
        --------
        galsim.GSObject
            PSF profile
        """
        if "fwhm" not in psf_params.keys():
                raise ValueError("fwhm parameter is required for psf. Add it to psf_params.")
        if psf_type=="moffat":
            if "beta" not in psf_params.keys():
                raise ValueError("beta parameter is required for moffat psf. Add it to psf_params.")
            psf = galsim.Moffat(
                beta=psf_params["beta"],
                fwhm=psf_params["fwhm"],
                gsparams=gsparams,
            )
        elif psf_type=="gaussian":
            psf = galsim.Gaussian(
                fwhm=psf_params["fwhm"],
                gsparams=gsparams,
            )
        else:
            raise ValueError("not yet implimented")

        return psf
    
    def simulate_galaxy_single_band(self, galaxy_params_filter, psf, add_noise, sky_level, filter, galaxy_type="sersic", gsparams=None):

        gal_flux = mag2counts(galaxy_params_filter["magnitude"], survey=self.survey, filter=filter)

        if galaxy_type=="sersic":
            galaxy = sim_single_band_sersic_galaxy(
                flux=gal_flux.value,
                hlr=galaxy_params_filter["hlr"],
                sersic_n=galaxy_params_filter["sersic_n"],
                axis_ratio=galaxy_params_filter["axis_ratio"],
                position_angle=galaxy_params_filter["position_angle"],
                gsparams=gsparams,
            )
        elif galaxy_type=="bulge+disk":
            raise ValueError("Not yet implemented")
        else:
            raise ValueError(f"galaxy_type should be either sersic or bulge+disk, got {galaxy_type}")

        # Convolve galaxy with PSF (both already have gsparams)
        gal_conv = galsim.Convolve([galaxy, psf], gsparams=gsparams)

        # Draw noiseless image (expected counts per pixel, used for ivar)
        image = gal_conv.drawImage(
            nx=self.image_size,
            ny=self.image_size,
            scale=self.survey.pixel_scale.to_value("arcsec"),
        )

        # Compute per-pixel inverse variance before adding noise.
        # For Poisson statistics, variance = expected counts.
        pixel_variance = np.zeros(image.array.shape, dtype=np.float32)

        if add_noise in ["galaxy", "all"]:
            galaxy_noise = galsim.PoissonNoise(rng=self.rng, sky_level=0.0)
            image.addNoise(galaxy_noise)
            pixel_variance += image.array

        if add_noise in ["background", "all"]:
            background_noise = galsim.PoissonNoise(rng=self.rng, sky_level=sky_level)
            noise_image = galsim.Image(self.image_size, self.image_size)
            noise_image.addNoise(background_noise)
            image += noise_image
            pixel_variance += sky_level

        return image, pixel_variance

    
    def simulate_galaxy(self, galaxy_params_multiband, add_noise=None, psf_type="moffat", psf_params_multiband={}):
        """
        Simulate multiband galaxy observations

        Parameters:
        -----------
        galaxy_params : dict
            Dictionary with filter names as keys and galaxy parameters as values.
            Each filter's parameters should be a dict for create_galaxy_profile.
            Example: {'g': {'magnitude': 22.0, 'half_light_radius': 0.3, ...},
                     'r': {'magnitude': 21.5, 'half_light_radius': 0.3, ...}}
        add_noise : bool
            Whether to add photon and sky noise

        Returns:
        --------
        dict of galsim.Image
            Dictionary of simulated galaxy images keyed by filter name
        """
        # Check if galaxy_params contains filter names (multiband mode)
        if add_noise is not None:
            if add_noise not in ["galaxy", "background", "all"]:
                raise ValueError(f"add_noise parameter should be either galaxy/background/all got {add_noise}")

        gsparams = self.gsparams

        psfs={}
        sky_levels={}
        for filter_name, psf_params in psf_params_multiband.items():
            if filter_name not in self.survey.available_filters:
                raise ValueError(f"Filter {filter_name} is not present in {self.survey}")
            psfs[filter_name] = self.get_psf(psf_type, psf_params, gsparams=gsparams)
            sky_levels[filter_name] = mean_sky_level(self.survey, self.survey.get_filter(filter_name)).to_value("electron")

        multi_band_image = {}
        multi_band_pixel_variance = {}

        for filter_name, band_params in galaxy_params_multiband.items():

            if filter_name not in self.survey.available_filters:
                raise ValueError(f"Filter '{filter_name}' not found in Survey")

            # Get filter-specific configuration
            filter = self.survey.get_filter(filter_name)

            image, pixel_variance = self.simulate_galaxy_single_band(
                galaxy_params_filter=band_params,
                psf=psfs[filter_name],
                add_noise=add_noise,
                sky_level=sky_levels[filter_name],
                filter=filter,
                gsparams=gsparams,
            )

            multi_band_image[filter_name] = image
            multi_band_pixel_variance[filter_name] = pixel_variance

        return multi_band_image, multi_band_pixel_variance


