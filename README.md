# Gaussian Mixture Model
This code primarily serves as a demo of the GaussianMixture class from the scikit-learn package in Python. Using astrometric data of Milky Way stars from the Gaia Archive, the GaussianMixture class offers a way to distinguish convolved Gaussian signals via velocities in the Local Standard of Rest frame.

Note: The code contains Python functions solely for data reduction -- separate from our main goal of deconvolution.

## Prerequisite Package Installation
Recommend making new Python environment, though your preferred environment will do just fine.

Install the required packages in a terminal via
```
> pip install requirements.txt
```
This will also install any dependent packages. For example, astroquery organizes data into astropy tables, so astropy will be installed as a dependency with astroquery.
