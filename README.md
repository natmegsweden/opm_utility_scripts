# opm_utility_scripts
 Utility scripts for Fieldline OPM system

plot_helmetscan.py plots a 2D layout showing which sensor was localized in which helmet slot (and which slots are empty). It is meant to allow the user to quickly see which slots are occupied by a working sensor and which aren't to help them assess whether or not it is needed to move sensors.

check_hpi.py checks a hpi recordings by fitting magentic dipoles to the detected coil fields. The function shows goodness of fits and plot the magnetic dipole locations along with the sensor locations. It is meant to help the user quickly ensure whether an hpi recording was successful. 

add_hpi_CP.py fits the hpi cois, calculates the device-to-head transform and then applies and adds it to a recording file. When executing the functions requires the user to select a data file (OPM-MEG recording that the transformation is applied to), a hpi file (OPM recording where hpi coils are activated), a polhemus file (TRIUX reccording containing a polhemus headshape), and a hpi frequency. 

add_hpi_multi_CP.py does the same but applies the transform to multiple recording files. Inputs are the same as in add_hpi_CP.py but the user can select multiple data files that the transformation is applied to.
