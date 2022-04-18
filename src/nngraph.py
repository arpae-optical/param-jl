from http.client import REQUESTED_RANGE_NOT_SATISFIABLE
import math
import random
from math import floor
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from matplotlib.collections import LineCollection
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import mean_squared_error
# from torch._C import UnionType
# from scipy import stat
import utils


def unnormalize(normed, min, max):
    return normed * (max - min) + min

def read_integral_emiss(filepath, index_str):
    real_array = []
    pred_array = []
    with open(filepath) as integral_file:
        lines = integral_file.readlines()
        real_array = np.array([float(line[0:7]) for line in lines])
        pred_array = np.array([float(line[9:]) for line in lines])
    print(len(pred_array))
    fig = plt.figure(1)
    s = [5 for n in range(len(pred_array))]
    plt.scatter(pred_array, real_array, alpha=0.003, s=s)
    plt.xlim([min(pred_array), max(pred_array)])
    plt.ylim([min(real_array), max(real_array)])
    r2 = r2_score(real_array, pred_array)
    plt.title(f"Laser Emissivity Points, Real vs Predicted, r^2 = {round(r2,4)}")
    plt.xlabel("Predicted Emissivities")
    plt.ylabel("Real Emissivites")
    real_array, pred_array = real_array.reshape(-1,1), pred_array.reshape(-1,1)
    
    plt.plot(real_array, LinearRegression().fit(real_array, pred_array).predict(real_array), color = "green", label = f"r-squared = {r2}",)
    # plt.annotate("r-squared = {:.3f}".format(r_value), (0, 1))
    plt.savefig(f"{index_str}.png")
    fig.clf()

def save_integral_emiss_point(predicted_emissivity, real_emissivity, filepath, wavelen_num = 300, all_points = False):
    
    wavelength = torch.load(Path("wavelength.pt"))
    wavelength = np.flip(np.array(wavelength.cpu())[0])
    eifile = open(filepath, "a")
    print("start file")
    for i_run_index in range(predicted_emissivity.size(dim = 0)):

        # Emiss residuals

        current_list = predicted_emissivity[i_run_index]

        real_emiss_list = real_emissivity[i_run_index]

        # old_emiss = predicted_emissivity[i_run_index][0:wavelen_num]
        # first_emiss = float(old_emiss[0])
        # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
        

        integral_real_total = 0
        integral_pred_total = 0
        for wavelen_i in range(wavelen_num-1):
            if all_points == False:
                integral_real_total += abs(float(real_emiss_list[wavelen_i])*float(wavelength[wavelen_i+1]-wavelength[wavelen_i]))
                integral_pred_total += abs(float(current_list[wavelen_i])*float(wavelength[wavelen_i+1]-wavelength[wavelen_i]))
            elif all_points == True:
                eifile.write(f"{float(real_emiss_list[wavelen_i]):.5f}, {float(current_list[wavelen_i]):.5f}\n")
            else:
                print("all_points must be True or False") #TODO: proper errors
        eifile.write(f"{float(integral_real_total):.5f}, {float(integral_pred_total):.5f}\n") #TODO Alok: return these two
    eifile.close()
    print("end file")


def emiss_error_graph(predicted_emissivity, real_emissivity, wavelen_num = 300):
    # Emiss residuals
    RMSE_total = 0
    MAPE_total = 0
    wavelength = torch.load(Path("wavelength.pt"))
    wavelength = np.flip(np.array(wavelength.cpu())[0])
    best_run_index = 0
    best_RMSE = 1
    worst_RMSE = 0
    RMSE_list = []
    worst_run_index = 0
    for i_run_index in range(50):

        # Emiss residuals

        current_list = predicted_emissivity[i_run_index]

        real_emiss_list = real_emissivity[i_run_index]

        # old_emiss = predicted_emissivity[i_run_index][0:wavelen_num]
        # first_emiss = float(old_emiss[0])
        # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
        
        MSE_E_P = 0
        for wavelen_i in range(wavelen_num):
            MSE_E_P += (real_emiss_list[wavelen_i] - current_list[wavelen_i]) ** 2
        RMSE_E_P = float(MSE_E_P / wavelen_num) ** (0.5)
        RMSE_total += RMSE_E_P/50

        RMSE_list.append(RMSE_E_P)

        if RMSE_E_P < best_RMSE:
            best_RMSE = RMSE_E_P
            best_run_index = i_run_index
        
        if RMSE_E_P > worst_RMSE:
            worst_RMSE = RMSE_E_P
            worst_run_index = i_run_index

        MAPE = 0
        for wavelen_i in range(wavelen_num):
            MAPE += abs(real_emiss_list[wavelen_i] - current_list[wavelen_i])
        MAPE = float(MAPE / wavelen_num)
        MAPE_total += MAPE/50
    RMSE_total = np.asarray(RMSE_total)
    average_run_index = (np.abs(RMSE_list - RMSE_total)).argmin()


    best_RMSE_pred = predicted_emissivity[best_run_index][0:wavelen_num]

    best_RMSE_real = real_emissivity[best_run_index][0:wavelen_num]

    worst_RMSE_pred = predicted_emissivity[worst_run_index][0:wavelen_num]

    worst_RMSE_real = real_emissivity[worst_run_index][0:wavelen_num]

    average_RMSE_pred = predicted_emissivity[average_run_index][0:wavelen_num]

    MSE_E_P = 0
    for wavelen_i in range(wavelen_num):
        MSE_E_P += (real_emissivity[average_run_index][wavelen_i] - predicted_emissivity[average_run_index][wavelen_i]) ** 2
    RMSE_average = float(MSE_E_P / wavelen_num) ** (0.5)

    average_RMSE_real = real_emissivity[average_run_index][0:wavelen_num]

    return([best_RMSE_pred, best_RMSE_real, worst_RMSE_pred, worst_RMSE_real, average_RMSE_pred, average_RMSE_real, wavelength, RMSE_total, RMSE_average])

def training_set_mean_vs_stdev():
    forward_train_data = torch.load("/data-new/alok/laser/data.pt")["interpolated_emissivity"]
    average_emiss = [[] for i in range(800)]
    e_total = 0
    for emiss_list in forward_train_data:
        for e_index, emiss in enumerate(emiss_list):
            e_total+=1
            average_emiss[e_index].append(float(emiss))
    mean_list = []
    stdev_list = []
    for a_index, emiss_list in enumerate(average_emiss):
        if a_index < 800:
            mean_list.append(np.mean(emiss_list))
            stdev_list.append(np.std(emiss_list))

    return(mean_list, stdev_list)

"""
pred_target: str
"""
def val_set_RMSE(pred_target = "/data-new/alok/laser/preds.pt"):

    mean_pred, trash = training_set_mean_vs_stdev()
    RMSE_total = 0
    mean_total = 0
    for i in range(5):
        all_preds = torch.load(pred_target)[i]
        real_emissivity = all_preds["true_emiss"]
        predicted_emissivity = all_preds["pred_emiss"]


        real = torch.load("/data-new/alok/laser/data.pt")
        wavelength = real["interpolated_wavelength"]


        for p in range(400):
            print(p)
            real_emiss_list = real_emissivity[p][0:800]
            predicted_emiss_list = predicted_emissivity[p][0:800]

            RMSE_total += mean_squared_error(real_emiss_list, predicted_emiss_list, squared = False)
            print(RMSE_total)
            mean_total += mean_squared_error(real_emiss_list, mean_pred, squared = False)
            print(mean_total)

    RMSE_total = RMSE_total/2000
    mean_total = mean_total/2000

    print(f"Predicted RMSE total: {RMSE_total}, mean vs real RMSE total: {mean_total}")
    

# y, stdevs = training_set_mean_vs_stdev()[0:300:15]
# wavelength = torch.load(Path("wavelength.pt"))
# x = np.flip(np.array(wavelength.cpu())[0])[0:300:15]
# plt.figure(111)
# s=[(n**2)*1000 for n in stdevs[0:300:15]]
# points = np.array([x, y]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)
# lc = LineCollection(segments, linewidths=s,color='blue')
# fig,a = plt.subplots()
# a.add_collection(lc)
# plt.savefig("test_graph_2.png")

def graph(residualsflag, predsvstrueflag, target_str, wavelen_num = 800, index_str="0"):
    # importing the data
    wavelength = np.linspace(2.5, 12.5, num=800)
    preds = torch.load(
        Path(f"{target_str}")
    )  # this'll just be str() if I end up not needing it
    preds = preds[int(index_str)]

    real = torch.load("/data-new/alok/laser/data.pt")
    real_laser = real["normalized_laser_params"]
    real_emissivity = preds["true_emiss"]

    predicted_emissivity = preds["pred_emiss"]

    # laser indexed [vae out of 50][wavelength out of wavelen_num][params, 14]
    predicted_laser = preds["params"]


    # extend emissivity wavelength x values
    plt.figure(1)       
    buckets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0-0.1, 0.1-0.2, ... , 0.9-1.0
    bucket_totals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if residualsflag == True:
        Emiss_E_P_list = []
        Laser_E_P_list = []
        for p in range(400):
            # index_array = [41,109,214,284,297,302,315,378]#,431,452
            # i_run_index = index_array[p]
            i_run_index = p

            # format the predicted params

            # sort by watt, then speed, then spacing

            plt.title("Emissivity vs Laser Param Scatter")
            plt.show()
            # temp = 1400
            # for i in range(159):
            
            val = real_laser[p].T.cpu()
            watt1 = val.T[2:]
            watt2 = np.where(watt1 == 1)
            watt = (watt2[0] + 2)/10
            watt = np.reshape(np.array(watt),(len(watt),1))

            speed = val.T[:1].T.cpu()

            spacing = val.T[1:2].T.cpu()


            predicted = predicted_laser[p].T.cpu()
            watt1 = predicted.T[2:]
            watt2 = np.where(watt1 == 1)
            watt_pred = (watt2[0] + 2)/10
            speed_pred = predicted.T[:1].T.cpu()

            spacing_pred = predicted.T[1:2].T.cpu()
            

            real_emiss_list = real_emissivity[p]
            predicted_emiss_list = predicted_emissivity[p]
            MSE_E_P = 0
            for wavelen_i in range(wavelen_num):
                MSE_E_P += (real_emiss_list[wavelen_i]-predicted_emiss_list[wavelen_i])**2
            
            RMSE_E_P = (MSE_E_P/wavelen_num)**(0.5)
            if len(watt) == 1 and len(watt_pred) == 1:
                watt_diff = (float(watt)-float(watt_pred))**2
                speed_diff = (float(speed)-float(speed_pred))**2
                space_diff = (float(spacing)-float(spacing_pred))**2
                RMSE_expected_predicted = ((watt_diff+speed_diff+space_diff)/3)**.5
                Laser_E_P_list.append(RMSE_expected_predicted)
                Emiss_E_P_list.append(RMSE_E_P)
                    
  
                    

        x = np.array(Laser_E_P_list)
        y = np.array(Emiss_E_P_list)
        x_len = len(x)
        # gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        # mn=np.min(x)
        # mx=np.max(x)
        # x1=np.linspace(mn,mx,500)
        # y1=gradient*x1+intercept
        plt.scatter(
                x,
                y,
                s=[30 for n in range(x_len)],
                alpha = 1,
                label="Real vs Predicted Emissivity vs Laser Param Residuals",
            )
        # plt.plot(x1,y1,'-r')
        rounded = str(round(np.mean(y),6))#needed to avoid rounding error
        print(rounded)
        plt.title("Laser Params vs Emiss")
        plt.xlabel(f"Laser Parameters Residuals")
        plt.ylabel(f"Emissivity Residuals (mean {rounded[0:8]})")
        # plt.annotate("r-squared = {:.3f}".format(r_value), (0, 1))
        plt.show()
        plt.savefig(f'{index_str}_emiss_laser_residual_graph.png')
        
        plt.clf()

    # randomly sample from real validation

    if predsvstrueflag == True:

        y, stdevs = training_set_mean_vs_stdev()
        wavelength = real["interpolated_wavelength"]
        x = np.array(wavelength.cpu())[0]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        stdevs = stdevs[:-1]
        lwidths=[dev*100 for dev in stdevs]
        for i_run_index in range(0, 500, 50):

            plt.figure(100 + i_run_index)
            RMSE_total = 0
            MAPE_total = 0
            emiss_arrays = [[] for i in range(800)]
            

            # Emiss residuals
            
            
            current_list = predicted_emissivity[i_run_index][0:wavelen_num]

            real_emiss_list = real_emissivity[i_run_index]

            old_emiss = predicted_emissivity[i_run_index]
            # first_emiss = float(old_emiss[0])
            # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
            
            MSE_E_P = 0
            for wavelen_i in range(wavelen_num):
                MSE_E_P += (
                    real_emiss_list[wavelen_i] - current_list[wavelen_i]
                ) ** 2
                
                emiss_arrays[wavelen_i].append(current_list[wavelen_i])
            RMSE_E_P = float(MSE_E_P / wavelen_num) ** (0.5)
            RMSE_total += RMSE_E_P

        
            lc = LineCollection(segments, linewidths=lwidths,color='blue',linestyles = "dotted")

            fig,a = plt.subplots()
            a.add_collection(lc)
            a.set_xlim(2.5,12)
            a.set_ylim(0,1.1)

            old_emiss = [np.mean(vae_array) for vae_array in emiss_arrays]
            # first_emiss = float(old_emiss[0])
            # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
            
            a.plot(
                wavelength[0:wavelen_num],
                old_emiss[0:wavelen_num],
                c="blue",
                alpha=0.1,
                linewidth=2.0,
                label=f"Predicted Emiss, average RMSE {round(RMSE_total/50,5)}, MAPE {round(MAPE_total/50,5)}",
            )

            new_emiss = real_emissivity[i_run_index]
            a.plot(
                wavelength[0:wavelen_num],
                new_emiss[0:wavelen_num],
                c="black",
                alpha=1,
                linewidth=2.0,
                label="Real Emissivity",
            )


            
            a.set_xlabel("Wavelength")
            a.set_ylabel("Emissivity")

            plt.savefig(f"{index_str}_vs_training_best_{i_run_index}.png", dpi=300)
            plt.close(fig)

def save_params(target_str, wavelen_num = 800):
    # importing the data
    wavelength = np.linspace(2.5, 12.5, num=800)
    pred_full = torch.load(
        Path(f"{target_str}")
    )  # this'll just be str() if I end up not needing it
    paramfile = open("pred_laser_params.txt", "a")
    paramfile.write("watt, speed, spacing\n")
    for i in range(5):
        
        preds = pred_full[i]

        real = torch.load("/data-new/alok/laser/data.pt")
        real_laser = real["normalized_laser_params"]

        # laser indexed [vae out of 50][wavelength out of wavelen_num][params, 14]
        predicted_laser = preds["params"]


        # extend emissivity wavelength x values
        plt.figure(1)       
        Emiss_E_P_list = []
        Laser_E_P_list = []
        for p in range(512):
            # index_array = [41,109,214,284,297,302,315,378]#,431,452
            # i_run_index = index_array[p]
            i_run_index = p

            # format the predicted params

            # sort by watt, then speed, then spacing

            plt.title("Emissivity vs Laser Param Scatter")
            plt.show()
            # temp = 1400
            # for i in range(159):
            

            predicted = predicted_laser[p].T.cpu()
            watt1 = predicted.T[2:]
            watt2 = np.where(watt1 == 1)
            watt_pred = (watt2[0] + 1)/10
            speed_pred = predicted.T[:1].T.cpu()

            spacing_pred = predicted.T[1:2].T.cpu()
            
            paramfile.write(f"{float(watt_pred):.5f}, {float(speed_pred):.5f}, {float(spacing_pred):.5f}\n")
    paramfile.close()
                


# for i in range(5):
#     graph(True, False, "/data-new/alok/laser/preds.pt", index_str=str(i))
            
# preds = torch.load(
#         Path(f"src/pred_iter_1_latent_size_43_k1_variance_0.2038055421837831")
#     )  # this'll just be str() if I end up not needing it
# preds = preds[0]
# real_laser = preds["true_params"]
# real_emissivity = preds["true_emiss"]
# predicted_emissivity = preds["pred_emiss"][1]
# save_integral_emiss_point(predicted_emissivity, real_emissivity, "test.txt")