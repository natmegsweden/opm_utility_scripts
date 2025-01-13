# opm_utility_scripts
 Utility scripts for Fieldline OPM system

plot_helmetscan.py plots a 2D layout showing which sensor was localized in which helmet slot (and which slots are empty).

check_hpi.py checks a hpi recordings by fitting magentic dipoles to the detected coil fields. The function shows goodness of fits and plot the magnetic dipole locations along with the sensor locations. 

add_hpi_CP.py fits the hpi cois, calculates the device-to-head transform and applies and adds it to a recording file. 

add_hpi_multi_CP.py does the same but applies the transform to multiple recording files. 
