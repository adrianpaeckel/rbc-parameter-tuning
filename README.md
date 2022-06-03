# RBC parameter tuning using Bayesian optimization

This is the repository containing all relevant files of my Master Thesis regarding the tuning of a rule-based controller (RBC) applied to an energy storage system. One can find the main files for the simulation framework organized in the battery/, demand/, pv/ and simulation/ folders. The files for data acquisition of the NEST building can be found in the acquisition/ folder. The files used to perform experiments via the OPC UA server are in the experiment/ folder. The scripts generating the figures are in the fig_scripts/ folder.

### Introduction
In this work, we use Gaussian Processes to build statistical models of a performance and constraint metrics as functions of the parameters of the RBC. In the first step, we perform a series of measurements on the system with some safe initial parameters of the RBC. Next, the metric values are derived from the measurements done on the system. We then use the derived values to update the models. Finally, we apply a safe exploration-exploitation policy to find the next candidates to optimal parameters based on the updated models. This optimization scheme falls into the family of Bayesian optimization (BO) algorithms.

Here we test two modifications of the BO (Srinivas et al., 2009) algorithm: safe bayesian optimization (SBO)(Berkenkamp et al., 2016) and contextual safe bayesian optimization (CSBO)(Berkenkamp et al., 2021). To this end, we build a simulation framework based on the historical data from the NEST building at EMPA. Two RBC serving different purposes are tested: the energy scheduling (ES) RBC and the peak-shaving (PS) RBC. Lastly, we perform a series of experiments with the ES RBC and CSBO at the Li-NMC battery in the same building.

### Setting up the environment
 The project is written in python 3.8.8, so first, you will need to set up your python environment. For this, you can use a package manager like conda (https://www.anaconda.com/) or **pip virtualenv**. To install all the necessary python packages you can use the /requirements.txt file provided here (**pip install -r requirements.txt**). Most of the scripts are written as Jupyter Notebooks, for this reason the easiest way to start running code is to install the Anaconda package. 
Here we use two python environments: one for general use (numerical simulations, figures, etc.) and a separate one for the experiments with the OPC UA server (use the opcua_requirements.txt file to build it).

### Get NEST data
First, we need to acquire all necessary data from the NEST building. For that, we can use the functions in dataqcui at mtfunc/ or use the acquisition script from the NEST repository https://gitlab.empa.ch/ues-lab/tools/nest/rest_client_python.git .

### Battery model
After storing the data we can start by building the model for the battery. Make sure there is a csv file in ES_model/PWLmodel_par_opti.csv (and .h5 files in case the ANN model is used) with the optimized parameters for the EPWL model of the battery. Otherwise run the /battery/model_optimization_NLopt.ipynb script. Afterwards run /battery/model_eval_2.ipynb to build and evaluate models.

### Load model
#### Demand model
Run all scripts in the demand/ folder, make sure the h5 files containing the resulting models are saved in saved_models/. Here we use CNNs to generate a 24h-ahead prediction of the demand power based on the demand of the past week and the temperature forecast for the next day. You can try other ANN architectures in the miscellaneous section at the end of demand/demand_1_2.ipynb. The hyperparameters characterizing the CNNs of both demand models (number of filters or neurons on each layer, training rate, etc.) can be optimized using the script at demand/hp_tuning/CNN_hp_tuning.py. You can modify this file to tune other sets of hyperparameters or other ANN architectures. Here we use the keras implementation of the Hyperband algorithm (Li, Lisha, et al. 2017).
#### PV model
Run all scripts in the pv/ folder, make sure the h5 files containing the resulting model are saved in saved_models/. Here we use again a CNN to model the power output of the PV system as a function of the direct global solar irradiance, temperature, humidity and solar angles (elevation and azimuth). 
#### Load scenario generator
After obtaining a model for the PV system and the demand power, it is time to generate a realistic load scenario in which we can perform the numerical evaluation. For this run the script in the load/ folder. Here we use real data from measurements at NEST as input for the models, however, one can obtain similar results using synthetic data. Make sure the generated load power data is saved in model_data/.

### Simulation
To perform the numerical evaluation, run all the scripts in the simulation/ folder in the following order:

- simulation/grid_mp.py with ES RBC uncommented (this runs for 2-4h depending on the processor)
- simulation/ESopt_mp.py 
- simulation/ESRBC_sim.ipynb
- simulation/grid_mp.py with PS RBC uncommented (this runs for 2-4h depending on the processor)
- simulation/PSopt_mp.py (only for reference purposes as the optimal parameters differ from simulation to simulation due to different initial SoC)
- simulation/PSRBC_sim.ipynb

You can avoid running the grid_mp.py by using the already optimized kernel hyperparameters provided in /simulation/hps/* . Feel free to experiment with different hyperparameters (smaller length scales and bigger variances result in more conservative hyperparameters).

#### Evaluation
You can use the scripts in fig_scripts to evaluate qualitatively the results of the simulations. For this, you will need to change some file paths with your newly created file paths.

### Experiment codes
To run an experiment in the battery system you will have to connect through the OPC UA server. For this, you can run the control script experiment/rbc_3_1_2.py. This script writes the battery power setpoints to the battery system, reads all necessary measurements and stores them in the storage folder experiment/ESRBC_exp0. 
I recommend using the shell script experiment/start_experiment.sh instead (bat file provided for Windows users). It restarts the script every time there is a connection error or similar. Another recommendation is to run the control script in a server located at EMPA or in a place with a stable internet connection.

The evaluation and optimization is performed in a separate script experiment/optimization.ipynb. Every day at 23:59 update your local experiment/ESRBC_exp0 folder with the remote folder (in case the control script is running from an EMPA server). Then run experiment/optimization.ipynb and update the new parameters in experiment/rbc_3_1_2.py.



### Base scripts
The scripts used to generate the results and other data used in the project are saved in the base_scripts folder. The scripts used above are commented and generalized versions based on the scripts in base_scripts. For specific examples, please take a look at this folder.





This project was supervised by Benjamin Huber (benjamin.huber@empa.ch) and Mohammad Khosravi (khosravm@control.ee.ethz.ch). The author is Adrian Paeckel (adrianpa@ethz.ch).