# Gaussian Mixture Model
This code primarily serves as a demo of the GaussianMixture class from the scikit-learn package in Python. Using astrometric data of Milky Way stars from the Gaia Archive, the GaussianMixture class offers a way to distinguish convolved Gaussian signals via velocities in the Local Standard of Rest (LSR) frame. The phase space signatures of the Thin Disk, Thick Disk, and Stellar Halo components of the Milky Way appear as Gaussians in the LSR frame, and deconvolving those signals offers a way to discern which component a given star likely belongs to.

Note: The code contains Python functions solely for data reduction -- separate from our main goal of deconvolution.

## Prerequisite Package Installation
You are recommended to make a new Python environment for organization purposes, though your preferred environment will do just fine.

In a Python terminal - e.g. Visual Studio Code - install the required packages using the 'requirements.txt' file:
```
> pip install requirements.txt
```
This will also install any dependent packages. For example, astroquery organizes data into astropy tables, so astropy will be installed as a dependency with astroquery.

## Example
In this scenario, a sample of 100,000 stars has been drawn. The sample has been cast through selection cuts, yielding a cut sample of 89,733 stars. The $V_{LSR}$ values for the stars has been computed and run through the GaussianMixture process. For demonstration purposes, the distribution has been fit to two Gaussian signals, but within the code (which aims to separate stellar components), three are used. A normalized histogram with 500 bins has been overlayed with the Gaussian signals.
![100k_stars_500_bins_GMM.pdf](https://github.com/jharry99/Gaussian-Mixture-Model-V1/files/9843135/100k_stars_500_bins_GMM.pdf)
The black curve is the sum of the constituent Gaussians, showing agreement between the fit distribution and the histogram -- though adding more Gaussians would dampen any disagreements. The means and standard deviations for each component are given in the legend.

To tweak the number of stars in the sample, adjust the "TOP \*" field in the Gaia query as desired.

To change the number of Gaussians that are being fit, see the n_components argument in the call of GaussianMixture.

## Extra Technical Info
The LSR frame is the reference frame of a star moving in a circular orbit around the Galactic Center (GC), at a radius equal to the distance from the Sun to the GC (usually denoted $R_0$).

With respect to the LSR frame, the orthogonal velocities in the directions radial to the GC, collinear with the direction of rotation, and perpendicular to
the galactic mid-plane are denoted $U_{LSR}$, $V_{LSR}$, and $W_{LSR}$, respectively.

