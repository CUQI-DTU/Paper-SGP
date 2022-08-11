# Paper-SGP
Code to reproduce results of article on Structural Gaussian Priors

# Main scripts
MLreconstruction.py is used to initialize the UQ algorithm. The initial guess is found by assuming no prior and running 20 CGLS iterations. Then the RMSE is computed for each iteration and the iteration with the lowest RMSE is chosen as the starting point for the UQ algorithm.

GaussianStructuralPrior_PrecSweep.py is used to estimate a good precision value for the GMRF prior. The script computes the RMSE for several GMRF precision values and the GMRF precision resulting in the lowest RMSE is chosen for the UQ algorithm. 

UQstructuralgaussianpriors.py is the main script where the SGP prior is applied and most figures in the paper are produced. 

# Tips for running the code
Make sure that the acquisition geometry (ag) is consistent in the script you are running and projectionfunctions.py

# Data
Synthetic data is found in the data-folder. It can be generated using the script GenerateSynthData.py
Real data is found ... something with zenodo?

# Phantom
For synthetic studies a pipe phantom is used. In the paper the DeepSeaOilPipe4 phantom from phantomlib.py is used.

