import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from astroquery.gaia import Gaia

# For rendering plot text in LaTeX font & adjusting font size
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams["font.size"] = "16"

### WARNING: This will take ~8 minutes to run (for 700k stars) ###

global R0
R0 = 8.178 # distance of the Sun from the galactic center, in kpc

def Find_LSR_Vels(numstars, ra_dec_data, l, b, distances, rad_vel):
    """
    This function calculates the LSR frame velocities of stars in the stellar sample, given their angular positions, distances, and heliocentric velocities
    
    Args:
        numstars (int): number of stars in the sample
        ra_dec_data (np.ndarray with shape (numstars, 4)): 2D array containing the ra, dec, pmra, and pmdec indexed columns, in deg and mas/yr respectively
        l (np.ndarray with shape (numstars, 1)): 1D array containing galactic longitude values, in deg
        b (np.ndarray with shape (numstars, 1)): 1D array containing galactic latitude values, in deg
        distances (np.ndarray with shape (numstars, 1)): 1D array containing distance values, in kpc
        rad_vel (np.ndarray with shape (numstars, 1)): 1D array containing radial velocity values, in km/s
    
    
    Returns:
        v_gal (np.ndarray with shape (numstars, 3)): 2D array containing U_LSR, V_LSR, and W_LSR velocity values, all in km/s
    """
    
    # Angular positions (RA and dec) of the galactic north pole
    alpha_g, delta_g = 192.85948*(np.pi/180), 27.12825*(np.pi/180) # converted from degrees to radians

    # Preculiar velocities of the Sun with respect to the LSR frame, all in km/s
    USun, VSun, WSun = 11.1, 12.24, 7.25

    converted_pm = np.zeros([numstars,2], dtype = float)
    tangent_vels = np.zeros([numstars,2], dtype = float)
    x = np.zeros([numstars,1], dtype = float)
    y = np.zeros([numstars,1], dtype = float)
    z = np.zeros([numstars,1], dtype = float)

    alpha = np.zeros([numstars,1], dtype = float)
    beta = np.zeros([numstars,1], dtype = float)
    
    v_xyz = np.zeros([numstars,3], dtype = float)
    v_gal = np.zeros([numstars,3], dtype = float)

    for i in range(numstars):
        C1 = np.sin(delta_g)*np.cos(ra_dec_data[i,1]*(np.pi/180))-np.cos(delta_g)*np.sin(ra_dec_data[i,1]*(np.pi/180))*np.cos(ra_dec_data[i,0]*(np.pi/180)-alpha_g)
        C2 = np.cos(delta_g)*np.sin(ra_dec_data[i,0]*(np.pi/180)-alpha_g)

        cosb = np.sqrt(pow(C1,2)+pow(C2,2))

        converted_pm[i,0] = (C1*ra_dec_data[i,2]+C2*ra_dec_data[i,3])/pow(cosb,2)  # mu_l, in rad/s
        converted_pm[i,1] = (C1*ra_dec_data[i,3]-C2*ra_dec_data[i,2])/cosb         # mu_b, in rad/s

        tangent_vels[i,0] = converted_pm[i,0]*distances[i]*np.cos(b[i]*(np.pi/180))  # v_l, in km/s
        tangent_vels[i,1] = converted_pm[i,1]*distances[i]  # v_b, in km/s

        v_xyz[i,0] = rad_vel[i]*np.cos(l[i]*(np.pi/180))*np.cos(b[i]*(np.pi/180))-tangent_vels[i,0]*np.sin(l[i]*(np.pi/180))-tangent_vels[i,1]*np.cos(l[i]*(np.pi/180))*np.sin(b[i]*(np.pi/180))    # U, in km/s
        v_xyz[i,1] = rad_vel[i]*np.sin(l[i]*(np.pi/180))*np.cos(b[i]*(np.pi/180))+tangent_vels[i,0]*np.cos(l[i]*(np.pi/180))-tangent_vels[i,1]*np.sin(l[i]*(np.pi/180))*np.sin(b[i]*(np.pi/180))    # V, in km/s
        v_xyz[i,2] = rad_vel[i]*np.sin(b[i]*(np.pi/180))+tangent_vels[i,1]*np.cos(b[i]*(np.pi/180))                                                                                                 # W, in km/s

        # galactocentric positions, in km
        x[i] = distances[i]*np.cos(b[i]*(np.pi/180))*np.cos(l[i]*(np.pi/180))-R0*3.08567758e16 
        y[i] = distances[i]*np.cos(b[i]*(np.pi/180))*np.sin(l[i]*(np.pi/180))
        z[i] = distances[i]*np.sin(b[i]*(np.pi/180))

        alpha[i] = np.arctan2(y[i],-1*x[i])                             # in radians
        beta[i] = np.arctan2(z[i],np.sqrt(pow(y[i],2)+pow(x[i],2)))     # in radians

        if alpha[i] < 0: # if angle < 0, add 2pi
            alpha[i] += 2*np.pi

        if beta[i] < 0: # if angle < 0, add 2pi
            beta[i] += 2*np.pi

        ShiftedU = v_xyz[i,0]+USun
        ShiftedV = v_xyz[i,1]+VSun
        ShiftedW = v_xyz[i,2]+WSun

        # Calculating the velocities in the LSR frame, via a spherical rotation
        v_gal[i,0] = np.cos(alpha[i])*np.cos(beta[i])*ShiftedU-np.sin(alpha[i])*np.cos(beta[i])*ShiftedV-np.sin(beta[i])*ShiftedW   # U_LSR, in km/s
        v_gal[i,1] = np.sin(alpha[i])*ShiftedU+np.cos(alpha[i])*ShiftedV                                                            # V_LSR, in km/s
        v_gal[i,2] = np.cos(alpha[i])*np.sin(beta[i])*ShiftedU-np.sin(alpha[i])*np.sin(beta[i])*ShiftedV+np.cos(beta[i])*ShiftedW   # W_LSR, in km/s
        
    return v_gal

def selection_cuts(dataset):
    """
    This function removes stars from a data set satisfying the following criteria:
    (1) Galactocentric R cut, R < 7 kpc & R > 9 kpc
    (2) Galactocentric phi cut, phi < 174 deg & phi > 186 deg
    (3) "Box" cuts of the Large & Small Magellanic Cloud sightlines, 
        (30 < |b| < 39 deg) and ((271 < l < 287 deg) or (73 < l < 89 deg))
        (41 < |b| < 48 deg) and ((299 < l < 307 deg) or (53 < l < 61 deg))
    
    Args:
        dataset (astropy.table.table.Table): input data table with index columns for distance 'd' in kpc, galactic 'l' in deg, galactic 'b' in deg, radial ascension 'ra' in deg, declination 'dec' in deg, 'radial_velocity' in km/s, proper motion in radial ascension direction 'pmra' in mas/yr, proper motion in declination direction 'pmdec' in mas/yr
    
    
    Returns:
        dataframe_cut_data (pandas.core.frame.Dataframe): output data table with same indexed columns as input data set, with appropriate rows excised
    """
    numstars = len(dataset)
    
    dataset = np.array(dataset[['d','l','b','ra','dec','radial_velocity','pmra','pmdec']])
    
    copy = np.array(dataset, copy=True)
    indices = np.array([], dtype = np.uint32)

    for i in range(numstars):
        flag = False
        x = dataset[i][0]*np.cos(dataset[i][2]*(np.pi/180))*np.cos(dataset[i][1]*(np.pi/180))-R0 # defined as < 0 towards the Sun from the GC
        y = dataset[i][0]*np.cos(dataset[i][2]*(np.pi/180))*np.sin(dataset[i][1]*(np.pi/180))
        z = dataset[i][0]*np.sin(dataset[i][2]*(np.pi/180))

        # Cuts in R
        R = np.sqrt(pow(x,2)+pow(y,2)+pow(z,2))
        if R < 7 or R > 9:
            flag = True

        # Cuts in Phi
        phi = np.arctan2(y,x) # arctan2 returns values -pi to pi, accounting for the right quadrant, but we want the range 0 to 2pi in Galactic Longitude

        if phi < 0: # if angle < 0, add 2pi
            phi += 2*np.pi

        phi *= (180/np.pi) # convert to degrees

        if phi < 174 or phi > 186:
            flag = True

        # "Box cuts" for LMC/SMC sightlines
        # Note reflection symmetry in the angles, so as to avoid sampling biases
        if (30 < np.abs(dataset[i][2]) < 39) and ((271 < dataset[i][1] < 287) or (73 < dataset[i][1] < 89)):
            flag = True

        if (41 < np.abs(dataset[i][2]) < 48) and ((299 < dataset[i][1] < 307) or (53 < dataset[i][1] < 61)):
            flag = True

        if flag == True:
            indices = np.append(indices, i)

    new_data = np.delete(copy, indices, axis=0)
    
    dataframe_cut_data = pd.DataFrame(new_data, columns = [r'd',r'l',r'b',r'ra',r'dec',r'radial_velocity',r'pmra',r'pmdec'])
    
    return dataframe_cut_data

# Grab 6D astrometric data from (the 3rd data release of) the Gaia data archive -- i.e. distance, angular position, velocities for each star
# The conditions after WHERE are partly referenced from Hinkel 2021 ("Axial Symmetry Tests of Milky Way Disk Stars Probe the Galaxy's Matter Distribution")
job = Gaia.launch_job_async("""
SELECT TOP 700000 1/gaia_source.parallax AS d, gaia_source.l, gaia_source.b, gaia_source.ra, gaia_source.dec, gaia_source.radial_velocity, gaia_source.pmra, gaia_source.pmdec FROM gaiadr3.gaia_source WHERE (gaia_source.radial_velocity IS NOT NULL AND ABS(gaia_source.b) > 30 AND gaia_source.phot_g_mean_mag > 14 AND gaia_source.phot_g_mean_mag < 18 AND gaia_source.bp_rp < 2.5 AND gaia_source.bp_rp > 0.5 AND ABS((1/gaia_source.parallax)*SIN(RADIANS(gaia_source.b))) < 3.0 AND ABS((1/gaia_source.parallax)*SIN(RADIANS(gaia_source.b))) > 0.2 AND (1/gaia_source.parallax)*COS(RADIANS(gaia_source.b)) < 1.2 AND gaia_source.parallax > 0 AND gaia_source.astrometric_params_solved = 31)
""")

query_data = job.get_results()

cut_data = selection_cuts(query_data)

numstars = len(cut_data)

ra_dec_data = cut_data[['ra','dec','pmra','pmdec']].to_numpy()
distances = cut_data[['d']].to_numpy()
l = cut_data[['l']].to_numpy()
b = cut_data[['b']].to_numpy()
rad_vel = cut_data[['radial_velocity']].to_numpy()

# Unit conversions
distances = distances*3.08567758e16 # Convert distances from kpc to km
ra_dec_data[:,2] = ra_dec_data[:,2]*1.53631e-16 # Convert pmra from mas/yr to radians/s
ra_dec_data[:,3] = ra_dec_data[:,3]*1.53631e-16 # Convert pmdec from mas/yr to radians/s

LSR_velocities = Find_LSR_Vels(numstars, ra_dec_data, l, b, distances, rad_vel)
V_LSR = LSR_velocities[:,1]

# Plot a histogram of LSR velocity in the V direction
plt.xlabel(r'$V_{LSR}$ (km/s)')
plt.ylabel(r'Normalized Counts')
plt.xlim(-250,100)

# Bin stars in V_LSR, normalized to total area 1 in the histogram
VHist = plt.hist(V_LSR, bins=3500, density=True)

# Calculate the midpoints of the histogram
Midpoints = 0.5*(VHist[1][1:]+VHist[1][:-1])

# Need to feed 2D array into the model function, so stack the midpoints of the histogram with the counts
Stacked_XY = np.stack((Midpoints, VHist[0]), axis=-1)

# Applying the GaussianMixture method to fit 3 Gaussian components to the data
GMM = GaussianMixture(n_components=3, max_iter=1000, random_state=0).fit(V_LSR.reshape(-1,1))

# Extract the means, variances, and weights of the deconvolved Gaussians from the GMM object
means = GMM.means_
covs = GMM.covariances_
weights = GMM.weights_

# Three Gaussian components are used for the thin disk/thick disk/stellar halo in this code, but in theory, can include more or less
x_axis = np.linspace(-250, 100, num=10000)
Gaussian1 = norm.pdf(x_axis, float(means[0]), np.sqrt(float(covs[0][0])))*weights[0]
Gaussian2 = norm.pdf(x_axis, float(means[1]), np.sqrt(float(covs[1][0])))*weights[1]
Gaussian3 = norm.pdf(x_axis, float(means[2]), np.sqrt(float(covs[2][0])))*weights[2]

plt.plot(x_axis, Gaussian1, color='orange', label='(Mean: '+str(round(float(means[0]),3))+', Std Dev: '+str(round(np.sqrt(float(covs[0][0])),3))+')')
plt.plot(x_axis, Gaussian2, color='red', label='(Mean: ' +str(round(float(means[1]),3))+', Std Dev: '+str(round(np.sqrt(float(covs[1][0])),3))+')')
plt.plot(x_axis, Gaussian3, color='#999999', label='(Mean: '+str(round(float(means[2]),3))+', Std Dev: '+str(round(np.sqrt(float(covs[2][0])),3))+')')
plt.plot(x_axis, Gaussian1+Gaussian2+Gaussian3, color='black')

plt.legend();
plt.show()