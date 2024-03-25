import numpy as np
import pandas as pd  
import scipy.stats as st
import json
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
from skl2onnx.common.data_types import FloatTensorType
from scipy.optimize import curve_fit
import onnxruntime as rt



# DATASETS

class Dataset:

    Jorgenson = pd.read_excel('datasets/Jorgenson_cpx_dat_MAL_PT_filtered.xlsx')
    ArcPl = pd.read_excel('datasets/ArcPl_filtered.xlsx')
    Unknowns = pd.read_excel('datasets/Unknowns-template.xlsx')


# STANDARD PARAMETERS

class Parameters:
    
    # Errors typically associated with each oxide measurement in the EPMA.

    ''' 
    # Relative errors (percentage. 3% for measurm. >1wt% and 8% for measurm. <1wt.%)
    'SiO2'   : 0.03,  
    'TiO2'   : 0.08,
    'Al2O3'  : 0.03,
    'FeO'    : 0.03,
    'MgO'    : 0.03,
    'MnO'    : 0.08,
    'CaO'    : 0.03,
    'Na2O'   : 0.08,
    'Cr2O3'  : 0.08,
    'k2O_Liq': 0.08,
    'water'  : 0.08
    '''
        
    oxide_rel_err = np.array([0.03,0.08,0.03,0.03,0.03,0.08,0.03,0.08,0.08])
    K_rel_err = np.array([0.08])
    water_rel_err = np.array([0.08])

    temperature_bins = [600,1050,1140,1250,1800]
    pressure_bins    = [0,5,10,15,20,25,30]
    
    

# PREPROCESSING

def data_imputation(dataFrame):
    dataFrame = dataFrame.fillna(0) 
    return dataFrame 

def replace_zeros(dataFrame, column_name):
    min_non_zero = dataFrame[dataFrame[column_name] > 0][column_name].min()
    epsilon = np.nextafter(0.0, 1.0)
    dataFrame.loc[dataFrame[column_name] == 0, column_name] = dataFrame[column_name].apply(
        lambda x: np.random.uniform(epsilon, min_non_zero) if x == 0 else x
    )
    return dataFrame 

#Pairwise Log-ratio transformation
def pwlt_transformation(X):
    X[X==0] = 0.001
    nr = len(X)  
    ni = len(X[0])
    nf = int(ni*(ni-1)/2)
    N = np.zeros((nr,nf))
    D = np.zeros((nr,nf))
    c_in = 0
    c_out = ni-1
    for i in range(0,ni-1):
        l = c_out-c_in
        N[:,c_in:c_out] = np.reshape(np.repeat(X[:,i], [l], axis=0),(-1,l))
        D[:,c_in:c_out] = X[:,i+1:]
        c_in = c_out
        c_out = c_out+(ni-i-2)
    return np.log(np.divide(N,D))


# DATA AGUMENTATION

def perturbation(X, y, std_dev_perc, n_perturbations=15):
    X_rep = np.repeat(X, repeats=n_perturbations, axis=0)
    y_rep = np.repeat(y, repeats=n_perturbations, axis=0)
    std_dev_rep = np.repeat([std_dev_perc], repeats=len(X_rep), axis=0)*X_rep
    np.random.seed(10)
    X_perturb = np.random.normal(X_rep, std_dev_rep)
    groups = np.repeat(np.arange(len(X)), repeats=n_perturbations, axis=0)
    return X_perturb, y_rep, groups

def input_perturbation(X, std_dev_perc, n_perturbations=15):
    X_rep = np.repeat(X, repeats=n_perturbations, axis=0)
    std_dev_rep = np.repeat([std_dev_perc], repeats=len(X_rep), axis=0)*X_rep
    np.random.seed(10)
    X_perturb = np.random.normal(X_rep, std_dev_rep)
    groups = np.repeat(np.arange(len(X)), repeats=n_perturbations, axis=0)
    return X_perturb, groups


# TRAIN-TEST SPLIT

# Split to manage unbalanced output
def balanced_train_test(X,y,bins,test_size,sample_names):
    
    out = np.digitize(y, bins, right = 1) 
    
    X_temp = np.concatenate((np.reshape(sample_names,(len(sample_names),1)),X),axis=1)
    X_train_temp, X_test_temp, y_train, y_test = train_test_split(X_temp ,y                                                                              ,test_size=test_size 
                                                                  ,random_state=42
                                                                  , stratify=out)
    X_train = X_train_temp[:,1:]
    X_test = X_test_temp[:,1:]
    train_index = X_train_temp[:,0].astype(int)
    test_index = X_test_temp[:,0].astype(int)
    return(X_train, X_test, y_train, y_test, train_index, test_index)


# HYPERPARAMETERS FINE-TUNING

# Extra-Trees: can manage unbalanced data as output and dependencies between the input samples generated via data augmentation
def ET_train_validation_balanced_perturbation(X_train
                                              ,y_train
                                              ,std_dev_perc
                                              ,bins
                                              ,k_fold=10
                                              ,n_estimators=200
                                              ,max_depth=15
                                              ,n_perturbations=15
                                              ,pwlt=False):
    X_train_perturb, y_train_perturb, groups = perturbation(X_train
                                                            ,y_train
                                                            ,std_dev_perc
                                                            ,n_perturbations)
    scaler = StandardScaler().fit(X_train_perturb)
    
    if pwlt:
        X_train_perturb = pwlt_transformation(X_train_perturb)
    else:
        X_train_perturb = scaler.transform(X_train_perturb)
        
    out = np.digitize(y_train_perturb, bins, right = 1)
    skf = StratifiedGroupKFold(n_splits=k_fold)
    skf.get_n_splits(X_train_perturb, out, groups)
    
    total_score = 0
    for i, (train_index, validation_index) in enumerate(skf.split(X_train_perturb, out, groups)):
        model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model_fit = model.fit(X_train_perturb[train_index],y_train_perturb[train_index])
        k_score = model.score(X_train_perturb[validation_index],y_train_perturb[validation_index])
        total_score += k_score
    return model,total_score/k_fold 

# Extra-Trees: can manage unbalanced data as output
def ET_train_validation_balanced(X_train
                                 ,y_train
                                 ,bins
                                 ,k_fold=10
                                 ,n_estimators=200
                                 ,max_depth=15
                                 ,pwlt=False):
    
    scaler = StandardScaler().fit(X_train)
    
    if pwlt:
        X_train = pwlt_transformation(X_train)
    else:
        X_train = scaler.transform(X_train)
    
    out = np.digitize(y_train, bins, right = 1)
    skf = StratifiedKFold(n_splits=k_fold)
    skf.get_n_splits(X_train, out)
    
    total_score = 0
    for i, (train_index, validation_index) in enumerate(skf.split(X_train, out)):
        model = ExtraTreesRegressor(n_estimators=n_estimators,max_depth=max_depth)
        model_fit = model.fit(X_train[train_index],y_train[train_index])
        k_score = model.score(X_train[validation_index],y_train[validation_index])
        total_score += k_score
    return model,total_score/k_fold 


# MODELS

# Bias correction function (Linear piecewise function)
def bias_f_line(x,a):
    return a*(x-x0)

def bias_f_temp(x,
                ang_left,popt_left,
                ang_right,popt_right):
    global x0 #I added this line!!!!!!!!!!!!!!!!!!!
    if x<ang_left:
        x0 = ang_left
        return bias_f_line(x,popt_left)
    elif x>ang_right:
        x0 = ang_right
        return bias_f_line(x,popt_right)
    return 0.0    

bias_f = np.vectorize(bias_f_temp)

def piecewise_line(bias_split,X_train,y_train,model):
    ang_left = (max(y_train)+(bias_split-1)*min(y_train))/bias_split
    ang_right = ((bias_split-1)*max(y_train)+min(y_train))/bias_split

    pred = model.predict(X_train)
    bias_pred = pred - y_train
     
    ind = np.arange(0,len(y_train),1)
    ind_I = ind[y_train<ang_left]
    ind_II = ind[y_train>ang_right]
    
    y_train_I = y_train[ind_I]
    bias_pred_I = bias_pred[ind_I]
    y_train_II = y_train[ind_II]
    bias_pred_II = bias_pred[ind_II]
    
    global x0
    x0 = ang_left
    popt_bias_I, pcov_bias_I = curve_fit(bias_f_line, y_train_I, bias_pred_I)
    x0 = ang_right
    popt_bias_II, pcov_bias_II = curve_fit(bias_f_line, y_train_II, bias_pred_II)
    
    return(popt_bias_I,popt_bias_II,ang_left,ang_right)

    
def P_T_predictors(output, model):

    if output == 'Pressure' and model == 'cpx_only':
        scaler=joblib.load("models/"+"Model_cpx_only_P_bias.joblib")
        predictor = rt.InferenceSession('models/'+'Model_cpx_only_P_bias'+'.onnx')
        with open("models/"+'Model_cpx_only_P_bias'+'.json') as json_file:
            bias_json = eval(json.load(json_file))
    
    elif output == 'Temperature' and model == 'cpx_only':
        scaler=joblib.load("models/"+"Model_cpx_only_T_bias.joblib")
        predictor = rt.InferenceSession("models/"+'Model_cpx_only_T_bias'+'.onnx')
        with open("models/"+'Model_cpx_only_T_bias'+'.json') as json_file:
            bias_json = eval(json.load(json_file))
    
    
    elif output == 'Pressure' and model == 'cpx_liquid':
        scaler=joblib.load("models/"+"Model_cpx_liquid_P_bias.joblib")
        predictor = rt.InferenceSession("models/"+'Model_cpx_liquid_P_bias'+'.onnx')
        with open("models/"+'Model_cpx_liquid_P_bias'+'.json') as json_file:
            bias_json = eval(json.load(json_file))
        
    elif output == 'Temperature' and model == 'cpx_liquid':
        scaler=joblib.load("models/"+"Model_cpx_liquid_T_bias.joblib")
        predictor = rt.InferenceSession("models/"+'Model_cpx_liquid_T_bias'+'.onnx')
        with open("models/"+'Model_cpx_liquid_T_bias'+'.json') as json_file:
            bias_json = eval(json.load(json_file))
            
    return(scaler, predictor, bias_json)
    
    

# EVALUATION

def max_perc(x):
    return(np.percentile(x,84))

def min_perc(x):
    return(np.percentile(x,16))




