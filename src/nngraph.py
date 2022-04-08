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
        real_array = np.array([float(line[0:7]) for line in lines[0:176000]])
        pred_array = np.array([float(line[9:]) for line in lines[0:176000]])
    print(len(pred_array))
    fig = plt.figure(1)
    s = [5 for n in range(len(pred_array))]
    plt.scatter(pred_array, real_array, alpha=0.003, s=s)
    plt.xlim([min(pred_array), max(pred_array)])
    plt.ylim([min(real_array), max(real_array)])
    r2 = r2_score(real_array, pred_array)
    plt.title(f"Laser Emiss Integral, Real vs Predicted, r^2 = {round(r2,4)}")
    plt.xlabel("Predicted Emiss Integral")
    plt.ylabel("Real Emiss Integral")
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


def graph(residualsflag, predsvstrueflag, target_str, wavelen_num = 300, index_str="default"):
    # importing the data
    wavelength = np.linspace(2.5, 12.5, num=300)
    preds = torch.load(
        Path(f"{target_str}")
    )  # this'll just be str() if I end up not needing it
    preds = preds[0]
    real_laser = preds["true_params"]
    real_emissivity = preds["true_emiss"]
    predicted_emissivity = preds["pred_emiss"]

    # laser indexed [vae out of 50][wavelength out of wavelen_num][params, 14]
    predicted_laser = preds["params"]
    
    arbitrary_index = 20


    # extend emissivity wavelength x values
    plt.figure(1)
    buckets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0-0.1, 0.1-0.2, ... , 0.9-1.0
    bucket_totals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if residualsflag == True:
        Emiss_E_P_list = []
        Laser_E_P_list = []
        for p in range(500):
            for vae_index in range (50):
                # index_array = [41,109,214,284,297,302,315,378]#,431,452
                # i_run_index = index_array[p]
                i_run_index = p

                # format the predicted params

                # sort by watt, then speed, then spacing

                plt.title("Emissivity vs Laser Param Scatter")
                plt.show()
                # temp = 1400
                # for i in range(159):
                
                val = real_laser[p]
                watt1 = val.T[4:]
                watt2 = np.where(watt1 == 1)
                watt = (watt2[0] + 2)/10
                watt = np.reshape(np.array(watt),(len(watt),1))

                speed = val.T[:1].T.cpu()
                speed = np.array(speed)

                spacing = val.T[1:2].T.cpu()
                spacing = np.array(spacing)


                predicted = predicted_laser[vae_index][p]
                watt1 = val.T[4:]
                watt2 = np.where(watt1 == 1)
                watt_pred = (watt2[0] + 2)/10

                speed = val.T[:1].T.cpu()
                speed_pred = np.array(speed)

                spacing = val.T[1:2].T.cpu()
                spacing_pred = np.array(spacing)
                


                real_emiss_list = real_emissivity[p]
                predicted_emiss_list = predicted_emissivity[vae_index][p]
                MSE_E_P = 0
                for wavelen_i in range(wavelen_num):
                    MSE_E_P += (real_emiss_list[wavelen_i]-predicted_emiss_list[wavelen_i])**2
                RMSE_E_P = (MSE_E_P/wavelen_num)**(0.5)
                if len(watt) == 1:
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
                s=[0.001 for n in range(x_len)],
                alpha = 0.1,
                label="Real vs Predicted Emissivity vs Laser Param Residuals",
            )
        # plt.plot(x1,y1,'-r')
        plt.title("Laser Params vs Emiss")
        plt.xlabel("Laser Parameters Residuals")
        plt.ylabel("Emissivity Residuals")
        # plt.annotate("r-squared = {:.3f}".format(r_value), (0, 1))
        plt.show()
        plt.savefig('temp_oops_expected_predicted_graph.png')
        
        plt.clf()

    # randomly sample from real validation

    if predsvstrueflag == True:
        for i_run_index in range(0, 50, 1):

            plt.figure(100 + i_run_index)
            RMSE_total = 0
            MAPE_total = 0
            for arbitrary_vae in range(50):
                print("vae: " + str(arbitrary_vae))

                # Emiss residuals

                current_list = predicted_emissivity[arbitrary_vae][i_run_index][0:wavelen_num]

                real_emiss_list = real_emissivity[i_run_index]

                old_emiss = predicted_emissivity[arbitrary_vae][i_run_index]
                # first_emiss = float(old_emiss[0])
                # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
                plt.plot(
                    wavelength[0:wavelen_num],
                    old_emiss[0:wavelen_num],
                    c="blue",
                    alpha=0.1,
                    linewidth=2.0,
                )
                MSE_E_P = 0
                for wavelen_i in range(wavelen_num):
                    MSE_E_P += (
                        real_emiss_list[wavelen_i] - current_list[wavelen_i]
                    ) ** 2
                RMSE_E_P = float(MSE_E_P / wavelen_num) ** (0.5)
                RMSE_total += RMSE_E_P

                MAPE = 0
                for wavelen_i in range(wavelen_num):
                    MAPE += abs(real_emiss_list[wavelen_i] - current_list[wavelen_i])
                MAPE = float(MAPE / wavelen_num)
                MAPE_total += MAPE

            old_emiss = predicted_emissivity[49][i_run_index]
            # first_emiss = float(old_emiss[0])
            # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
            plt.scatter(
                wavelength[0:wavelen_num],
                [0 for n in range(wavelen_num)],
                s=[0.001 for n in range(wavelen_num)],
                label="Point density for reference",
            )
            plt.plot(
                wavelength[0:wavelen_num],
                old_emiss[0:wavelen_num],
                c="blue",
                alpha=0.1,
                linewidth=2.0,
                label=f"Predicted Emiss, average RMSE {round(RMSE_total/50,5)}, MAPE {round(MAPE_total/50,5)}",
            )

            new_emiss = real_emissivity[i_run_index]
            plt.plot(
                wavelength[0:wavelen_num],
                new_emiss[0:wavelen_num],
                c="black",
                alpha=1,
                linewidth=2.0,
                label="Real Emissivity",
            )

            leg = plt.legend(loc="upper right", prop={"size": 8})

            leg.legendHandles[0].set_alpha(1)
            plt.xlabel("Wavelength")
            plt.ylabel("Emissivity")

            plt.savefig(f"{index_str}_vs_training_best_{i_run_index}.png", dpi=300)
            plt.clf()
# preds = torch.load(
#         Path(f"src/pred_iter_1_latent_size_43_k1_variance_0.2038055421837831")
#     )  # this'll just be str() if I end up not needing it
# preds = preds[0]
# real_laser = preds["true_params"]
# real_emissivity = preds["true_emiss"]
# predicted_emissivity = preds["pred_emiss"][1]
# save_integral_emiss_point(predicted_emissivity, real_emissivity, "test.txt")