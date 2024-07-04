# Enhancing-ML-Thermobarometry-for-Clinopyroxene-Bearing-Magmas

Hello and welcome!

This is the series of codes used to calibrate and test the different results from the enhanced clinopyroxene (cpx) and clinoyroxene-liquid (cpx-liq) thermo-barometers presented in the manuscript "Enhancing Machine Learning Thermobarometry for Clinopyroxene-Bearing Magmas", submitted to the Computers and Geosciences journal.

link to the web app: https://cpx-thermobarometry.streamlit.app

Inside the folder entitled "ML_PT_Pyworkflow", you will find: 

    1.requirements.txt: A list of all the packages you need to run the module. Once placed in the "ML_PT_Pyworkflow" folder, you can run the line pip install -r requirements.txt on your terminal. You need to have installed previously pip and Python (version=3.11.5) in your environment.
    
    2.ML_PT_Pyworkflow.py: This is our Python module, which contains all the functions used throughout the workflow. You do not need to touch or change anything from here.
    
    3.Single_OutPut_Model.ipynb: This script contains all the steps used to train, validate and test our models. The optimized models are already saved in the folder "models", so you do not need to run this script if you only want to apply the models to retrieve P and T estimates.
    
    4.datasets folder: This folder contains the datasets we used through the workflow. You can modify the "Unknowns-template" and add your own values.
    
    5.Single_OutPut_Predictor.ipynb: This is the ready-to-apply script. You just need to open it (run the line Jupyter lab in your terminal, and a new tab will open in your navigator), and then you can customize the settings (if you want to use a "cpx_only" or "cpx_liquid" model and if you want to estimate "Temperature" or "Pressure"). Run it and get your estimates!
    
    6.models folder: Contains the pre-trained models.
    
    7.out-files folder: Once you run the "Single_OutPut_Predictor.ipynb" script, you will find your predictions here.
    
    8.Figures folder: Once you run the "Single_OutPut_Predictor.ipynb" script, you will find your figures here.

If you have any issues, please write to: maurizio.petrelli@unipg.it, monica.agredalopez@dottorandi.unipg.it

Distributed under the MIT license.
