# -*- coding: utf-8 -*-

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.base import clone
from osgeo import gdal

import xgboost as xgb
import catboost
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm

# Suppress warnings
warnings.filterwarnings("ignore")

# Define paths
in_path = r"#train_model_path"
out_path1 = in_path

# Define functions and models
def timemodel(COSSZA, kNDVI, LST_Day, LST_night, NDVI, NDWI, NIRv, FPAR, SM, VPD, DEM, slope, aspect):
    # Load training and testing data
    X_train = np.array(pd.read_excel(os.path.join(in_path, "trainx.xlsx"), index_col=0))
    y_train = np.array(pd.read_excel(os.path.join(in_path, "trainy.xlsx"), index_col=0))
    X_test = np.array(pd.read_excel(os.path.join(in_path, "testx.xlsx"), index_col=0))
    y_test = np.array(pd.read_excel(os.path.join(in_path, "testy.xlsx"), index_col=0))

    # Random Forest regressor model
    cb1 = RandomForestRegressor()
    cb1.fit(X_train, y_train)
    selection = SelectFromModel(cb1, prefit=True)
    select_X_train = selection.transform(X_train)
    selection_model1 = RandomForestRegressor()
    selection_model1.fit(select_X_train, y_train)
    y_pred_cb1 = selection_model1.predict(select_X_train)
    cb1_rmse = mean_squared_error(y_train, y_pred_cb1, squared=False)

    # CatBoost regressor model
    cb2 = catboost.CatBoostRegressor(loss_function='RMSE', max_depth=4, random_seed=47)
    cb2.fit(X_train, y_train)
    selection = SelectFromModel(cb2, threshold="median", prefit=True)
    select_X_train = selection.transform(X_train)
    selection_model2 = catboost.CatBoostRegressor(loss_function='RMSE', max_depth=4, random_seed=47)
    selection_model2.fit(select_X_train, y_train)
    y_pred_cb2 = selection_model2.predict(select_X_train)
    cb2_rmse = mean_squared_error(y_train, y_pred_cb2, squared=False)

    # Gradient Boosting regressor model
    cb3 = ensemble.GradientBoostingRegressor()
    cb3.fit(X_train, y_train)
    selection = SelectFromModel(cb3, threshold="median", prefit=True)
    select_X_train = selection.transform(X_train)
    selection_model3 = RandomForestRegressor()
    selection_model3.fit(select_X_train, y_train)
    y_pred_cb3 = selection_model3.predict(select_X_train)
    cb3_rmse = mean_squared_error(y_train, y_pred_cb3, squared=False)

    # Compute weights
    weights = [1 / cb1_rmse, 1 / cb2_rmse, 1 / cb3_rmse]
    weights_sum = sum(weights)
    weights = [w / weights_sum for w in weights]

    # Create stacked ensemble by manually averaging predictions with weights
    y_pred_stacked = (weights[0] * y_pred_cb1 +
                      weights[1] * y_pred_cb2 +
                      weights[2] * y_pred_cb3)
    stacked_rmse = mean_squared_error(y_train, y_pred_stacked, squared=False)

    # Train final estimator on stacked ensemble predictions
    final_estimator = LinearRegression()

    # Define Weighted Stacking Regressor class
    class WeightedStackingRegressor:
        def __init__(self, base_models, final_estimator, weights):
            self.base_models = base_models
            self.final_estimator = final_estimator
            self.weights = weights

        def fit(self, X, y):
            self.base_models_ = [clone(model) for model in self.base_models]
            for model in self.base_models_:
                model.fit(X, y)
            return self

        def predict(self, X):
            predictions = np.column_stack([
                model.predict(X) for model in self.base_models_
            ])
            weighted_predictions = np.average(predictions, axis=1, weights=self.weights)
            X_stacked = np.column_stack((X, weighted_predictions.reshape(-1, 1)))

            # Train final estimator on stacked predictions
            X_train_predictions = np.column_stack([
                model.predict(X_train) for model in self.base_models_
            ])
            X_train_weighted_predictions = np.average(X_train_predictions, axis=1, weights=self.weights)
            X_train_stacked = np.column_stack((X_train, X_train_weighted_predictions.reshape(-1, 1)))
            final_estimator.fit(X_train_stacked, y_train)

            # Return predictions on stacked features
            return self.final_estimator.predict(X_stacked)

    # Create weighted stacking regressor
    stacked_model = WeightedStackingRegressor(
        base_models=[selection_model1, selection_model2, selection_model3],
        final_estimator=final_estimator,
        weights=weights
    )

    # Fit stacked model
    stacked_model.fit(X_train, y_train)

    # Make predictions with the stacked model
    y_pred3 = stacked_model.predict(X_test)
    R2 = 1 - (np.sum((y_test - y_pred3) ** 2)) / (np.sum((y_test - y_test.mean()) ** 2))
    rmse = np.sqrt(np.mean((y_test - y_pred3) ** 2))
    aa, bb = stats.pearsonr(y_test.flatten(), y_pred3.flatten())
    mse = np.mean(np.square(y_pred3 - y_test))
    bias = np.mean(np.abs(y_pred3 - y_test))
    SIF = stacked_model.predict(np.hstack([COSSZA, kNDVI, LST_Day, LST_night, NDVI, NDWI, NIRv, FPAR, SM, VPD, DEM,
                                           slope, aspect]))
    return SIF

def replaceNaN(datMat):
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        datMat[nonzero(isnan(datMat[:,i]))[0],i] = 0
    return datMat

# Define paths for input rasters
COSSZApath = r'#COSSZA_path'
kNDVIpath = r'#kNDVI_path'
LSTDaypath = r'#LST_Day_path'
LSTnightpath = r'#LST_night_path'
NDVIpath = r'#NDVI_path'
NDWIpath = r'#NDWI_path'
NIRvpath = r'#NIRv_path'
FPARpath = r'#FPAR_path'
VPDpath = r'#VPD_path'
SMpath = r'#SM_path'
DEMpath = r'#DEM_path'
aspectpath = r'#aspect_path'
slopepath = r'#slope_path'
outroot = r'#outroot'

# Create output directory if it doesn't exist
if not os.path.exists(outroot):
    os.makedirs(outroot)

# Run model for specified years and dates
years = ['2022']
date_list = []
for date in range(60, 300, 8):
    date_list.append(date)

for year in years:
    for date in date_list:
        if int(date) < 100:
            date = "0" + str(date)
        ras_cossza = os.path.join(COSSZApath, year, str(year) + str(date) + ".tif")
        if os.path.exists(ras_cossza):
            rasCOSSZA = glob.glob(os.path.join(COSSZApath, year, str(year) + str(date) + ".tif"))
            raskNDVI = glob.glob(os.path.join(kNDVIpath, year, str(year) + str(date) + ".tif"))
            rasLSTDay = glob.glob(os.path.join(LSTDaypath, year, str(year) + str(date) + ".tif"))
            rasLSTnight = glob.glob(os.path.join(LSTnightpath, year, str(year) + str(date) + ".tif"))
            rasNDVI = glob.glob(os.path.join(NDVIpath, year, str(year) + str(date) + ".tif"))
            rasNDWI = glob.glob(os.path.join(NDWIpath, year, str(year) + str(date) + ".tif"))
            rasNIRv = glob.glob(os.path.join(NIRvpath, year, str(year) + str(date) + ".tif"))
            rasFPAR = glob.glob(os.path.join(FPARpath, year, str(year) + str(date) + ".tif"))
            rasSM = glob.glob(os.path.join(SMpath, year, str(year) + str(date) + ".tif"))
            rasVPD = glob.glob(os.path.join(VPDpath, year, str(year) + str(date) + ".tif"))
            rasDEM = glob.glob(os.path.join(DEMpath, "vation.tif"))
            rasslope = glob.glob(os.path.join(slopepath, "slope.tif"))
            rasaspect = glob.glob(os.path.join(aspectpath, "aspect.tif"))
            variables = [rasCOSSZA, raskNDVI, rasLSTDay, rasLSTnight, rasNDVI, rasNDWI, rasNIRv, rasFPAR, rasSM,
                         rasVPD, rasDEM, rasslope, rasaspect]

            # Check if all variables are not None
            if all(variable[0] is not None for variable in variables):
                COSSZA = gdal.Open(rasCOSSZA[0])
                kNDVI = gdal.Open(raskNDVI[0])
                LST_Day = gdal.Open(rasLSTDay[0])
                LST_night = gdal.Open(rasLSTnight[0])
                NDVI = gdal.Open(rasNDVI[0])
                NDWI = gdal.Open(rasNDWI[0])
                NIRv = gdal.Open(rasNIRv[0])
                FPAR = gdal.Open(rasFPAR[0])
                SM = gdal.Open(rasSM[0])
                VPD = gdal.Open(rasVPD[0])
                DEM = gdal.Open(rasDEM[0])
                slope = gdal.Open(rasslope[0])
                aspect = gdal.Open(rasaspect[0])

                # Read raster data as arrays
                COSSZA_array = COSSZA.ReadAsArray().reshape(-1, 1)
                kNDVI_array = kNDVI.ReadAsArray().reshape(-1, 1)
                LST_Day_array = LST_Day.ReadAsArray().reshape(-1, 1)
                LST_night_array = LST_night.ReadAsArray().reshape(-1, 1)
                NDVI_array = NDVI.ReadAsArray().reshape(-1, 1)
                NDWI_array = NDWI.ReadAsArray().reshape(-1, 1)
                NIRv_array = NIRv.ReadAsArray().reshape(-1, 1)
                FPAR_array = FPAR.ReadAsArray().reshape(-1, 1)
                SM_array = SM.ReadAsArray().reshape(-1, 1)
                VPD_array = VPD.ReadAsArray().reshape(-1, 1)
                DEM_array = DEM.ReadAsArray().reshape(-1, 1)
                slope_array = slope.ReadAsArray().reshape(-1, 1)
                aspect_array = aspect.ReadAsArray().reshape(-1, 1)

                # Ensure replacing NaN values
                COSSZA_array = replaceNaN(COSSZA_array)
                kNDVI_array = replaceNaN(kNDVI_array)
                LST_Day_array = replaceNaN(LST_Day_array)
                LST_night_array = replaceNaN(LST_night_array)
                NDVI_array = replaceNaN(NDVI_array)
                NDWI_array = replaceNaN(NDWI_array)
                NIRv_array = replaceNaN(NIRv_array)
                FPAR_array = replaceNaN(FPAR_array)
                SM_array = replaceNaN(SM_array)
                VPD_array = replaceNaN(VPD_array)
                DEM_array = replaceNaN(DEM_array)
                slope_array = replaceNaN(slope_array)
                aspect_array = replaceNaN(aspect_array)

                # Predict SIF using the model
                SIF = timemodel(COSSZA_array, kNDVI_array, LST_Day_array, LST_night_array, NDVI_array, NDWI_array,
                                NIRv_array, FPAR_array, SM_array, VPD_array, DEM_array, slope_array, aspect_array)
                # Reshape SIF array
                SIF = np.array(SIF).reshape(10158, 13712)
                # Define output directory and file name
                outws = os.path.join(outroot, year)
                if not os.path.exists(outws):
                    os.makedirs(outws)
                outname = os.path.join(outws, "SIF" + str(year) + str(date) + '.tif')

                # Create output GeoTIFF file
                gtiff_driver = gdal.GetDriverByName('GTiff')
                out_ds = gtiff_driver.Create(outname, 13712, 10518, 1, gdal.GDT_Float32)
                out_ds.SetProjection(NDWI.GetProjection())
                out_ds.SetGeoTransform(NDWI.GetGeoTransform())
                out_ds.GetRasterBand(1).WriteArray(SIF)
                out_ds.FlushCache()
                print(outname + " has been created")
            else:
                # Print message when at least one variable is None
                print("At least one variable is None.")
