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
        real_array = np.array([float(line[0:7]) for line in lines[0:3000]])
        pred_array = np.array([float(line[9:]) for line in lines[0:3000]])

    fig = plt.figure(1)
    s = [40 for n in range(len(pred_array))]
    plt.scatter(pred_array, real_array, alpha=0.03, s=s)
    plt.xlim([0.5, 3])
    plt.ylim([0.5, 3])
    r2 = r2_score(real_array, pred_array)
    plt.title(f"Laser Emiss Integral, Real vs Predicted, r^2 = {round(r2,4)}")
    plt.xlabel("Predicted Emiss Integral")
    plt.ylabel("Real Emiss Integral")
    real_array, pred_array = real_array.reshape(-1,1), pred_array.reshape(-1,1)
    
    plt.plot(real_array, LinearRegression().fit(real_array, pred_array).predict(real_array), color = "green", label = f"r-squared = {r2}",)
    # plt.annotate("r-squared = {:.3f}".format(r_value), (0, 1))
    plt.savefig(f"{index_str}_graph_buckets.png")
    fig.clf()


def save_integral_emiss_point(predicted_emissivity, real_emissivity, filepath):
    
    wavelength = torch.load(Path("wavelength.pt"))
    wavelength = np.flip(np.array(wavelength.cpu())[0])
    eifile = open(filepath, "a")
    print("start file")
    for i_run_index in range(predicted_emissivity.size(dim = 0)):

        # Emiss residuals

        current_list = predicted_emissivity[i_run_index]

        real_emiss_list = real_emissivity[i_run_index]

        # old_emiss = predicted_emissivity[i_run_index][0:518]
        # first_emiss = float(old_emiss[0])
        # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
        

        integral_real_total = 0
        integral_pred_total = 0
        for wavelen_i in range(518):
            integral_real_total += abs(float(real_emiss_list[wavelen_i])*float(wavelength[wavelen_i+1]-wavelength[wavelen_i]))
            integral_pred_total += abs(float(current_list[wavelen_i])*float(wavelength[wavelen_i+1]-wavelength[wavelen_i]))
        eifile.write(f"{float(integral_real_total):.5f}, {float(integral_pred_total):.5f}\n")
    eifile.close()
    print("end file")


def emiss_error_graph(predicted_emissivity, real_emissivity):
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

        # old_emiss = predicted_emissivity[i_run_index][0:518]
        # first_emiss = float(old_emiss[0])
        # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
        
        MSE_E_P = 0
        for wavelen_i in range(518):
            MSE_E_P += (real_emiss_list[wavelen_i] - current_list[wavelen_i]) ** 2
        RMSE_E_P = float(MSE_E_P / 518) ** (0.5)
        RMSE_total += RMSE_E_P/50

        RMSE_list.append(RMSE_E_P)

        if RMSE_E_P < best_RMSE:
            best_RMSE = RMSE_E_P
            best_run_index = i_run_index
        
        if RMSE_E_P > worst_RMSE:
            worst_RMSE = RMSE_E_P
            worst_run_index = i_run_index

        MAPE = 0
        for wavelen_i in range(518):
            MAPE += abs(real_emiss_list[wavelen_i] - current_list[wavelen_i])
        MAPE = float(MAPE / 518)
        MAPE_total += MAPE/50
    RMSE_total = np.asarray(RMSE_total)
    average_run_index = (np.abs(RMSE_list - RMSE_total)).argmin()


    best_RMSE_pred = predicted_emissivity[best_run_index][0:518]

    best_RMSE_real = real_emissivity[best_run_index][0:518]

    worst_RMSE_pred = predicted_emissivity[worst_run_index][0:518]

    worst_RMSE_real = real_emissivity[worst_run_index][0:518]

    average_RMSE_pred = predicted_emissivity[average_run_index][0:518]

    MSE_E_P = 0
    for wavelen_i in range(518):
        MSE_E_P += (real_emissivity[average_run_index][wavelen_i] - predicted_emissivity[average_run_index][wavelen_i]) ** 2
    RMSE_average = float(MSE_E_P / 518) ** (0.5)

    average_RMSE_real = real_emissivity[average_run_index][0:518]

    return([best_RMSE_pred, best_RMSE_real, worst_RMSE_pred, worst_RMSE_real, average_RMSE_pred, average_RMSE_real, wavelength, RMSE_total, RMSE_average])


def graph(residualsflag, predsvstrueflag, index_str="default", target_str="0"):
    # importing the data
    wavelength = torch.load(Path("wavelength.pt"))
    wavelength = np.flip(np.array(wavelength.cpu())[0])
    preds = torch.load(
        Path(f"{target_str}")
    )  # this'll just be str() if I end up not needing it
    preds = preds[0]
    print(preds)
    real_laser = preds["true_params"]
    real_emissivity = preds["true_emiss"]
    predicted_emissivity = preds["pred_emiss"]

    # laser indexed [vae out of 50][wavelength out of 518][params, 14]
    predicted_laser = preds["params"]

    arbitrary_index = 20

    # splits = split(len(predicted_laser))
    temp_predicted_laser = predicted_laser

    val_real_emissivity = real_emissivity

    # splits = split(len(predicted_emissivity))
    val_predicted_emissivity = predicted_emissivity

    # extend emissivity wavelength x values
    extended_max = 2.5
    extended_min = 0.5
    granularity = 100

    plt.figure(1)
    buckets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0-0.1, 0.1-0.2, ... , 0.9-1.0
    bucket_totals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if residualsflag == True:
        MSE_list = []
        params_list = []
        integral_pred_array = []
        integral_real_array = []
        for p in range(0, 500):
            print(p)
            # index_array = [41,109,214,284,297,302,315,378]#,431,452
            # i_run_index = index_array[p]
            i_run_index = p

            # format the predicted params

            # sort by watt, then speed, then spacing

            plt.title("Predicted Emissivity vs Ideal TPV Emitter")
            plt.show()
            # temp = 1400
            for arbitrary_vae in range(50):
                new_score = 0

                # format the predicted params

                real_emiss_list = real_emissivity[i_run_index]
                predicted_emiss_list = predicted_emissivity


                # Emiss residuals
                j = arbitrary_vae
                current_list = predicted_emiss_list[j][i_run_index][0:518]

                integral_real_total = 0
                integral_pred_total = 0
                for wavelen_i in range(518):
                    integral_real_total += abs(real_emiss_list[wavelen_i]*(wavelength[wavelen_i+1]-wavelength[wavelen_i]))
                    integral_pred_total += abs(current_list[wavelen_i]*(wavelength[wavelen_i+1]-wavelength[wavelen_i]))
                integral_pred_array.append(integral_pred_total)
                integral_real_array.append(integral_real_total)


                
        fig = plt.figure(1)
        s = [40 for n in range(len(integral_pred_array))]
        plt.scatter(integral_pred_array, integral_real_array, alpha=0.03, s=s)
        plt.xlim([0.5, 2.5])
        plt.ylim([0.5, 2.5])
        plt.title("Laser Emiss Integral, Real vs Predicted")
        plt.xlabel("Predicted Emiss Integral")
        plt.ylabel("Real Emiss Integral")
        # plt.annotate("r-squared = {:.3f}".format(r_value), (0, 1))
        plt.savefig(f"{index_str}_graph_buckets.png")
        fig.clf()

    # randomly sample from real validation

    if predsvstrueflag == True:
        for i_run_index in range(0, 50, 1):

            plt.figure(100 + i_run_index)
            RMSE_total = 0
            MAPE_total = 0
            for arbitrary_vae in range(50):
                print("vae: " + str(arbitrary_vae))

                # Emiss residuals

                current_list = predicted_emissivity[arbitrary_vae][i_run_index][0:518]

                real_emiss_list = real_emissivity[i_run_index]

                old_emiss = predicted_emissivity[arbitrary_vae][i_run_index]
                # first_emiss = float(old_emiss[0])
                # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
                plt.plot(
                    wavelength[0:518],
                    old_emiss[0:518],
                    c="blue",
                    alpha=0.1,
                    linewidth=2.0,
                )
                MSE_E_P = 0
                for wavelen_i in range(518):
                    MSE_E_P += (
                        real_emiss_list[wavelen_i] - current_list[wavelen_i]
                    ) ** 2
                RMSE_E_P = float(MSE_E_P / 518) ** (0.5)
                RMSE_total += RMSE_E_P

                MAPE = 0
                for wavelen_i in range(518):
                    MAPE += abs(real_emiss_list[wavelen_i] - current_list[wavelen_i])
                MAPE = float(MAPE / 518)
                MAPE_total += MAPE

            old_emiss = predicted_emissivity[49][i_run_index]
            # first_emiss = float(old_emiss[0])
            # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
            plt.scatter(
                wavelength[0:518],
                [0 for n in range(518)],
                s=[0.001 for n in range(518)],
                label="Point density for reference",
            )
            plt.plot(
                wavelength[0:518],
                old_emiss[0:518],
                c="blue",
                alpha=0.1,
                linewidth=2.0,
                label=f"Predicted Emiss, average RMSE {round(RMSE_total/50,5)}, MAPE {round(MAPE_total/50,5)}",
            )

            new_emiss = real_emissivity[i_run_index]
            plt.plot(
                wavelength[0:518],
                new_emiss[0:518],
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
# preds = torch.load(
#         Path(f"src/pred_iter_1_latent_size_43_k1_variance_0.2038055421837831")
#     )  # this'll just be str() if I end up not needing it
# preds = preds[0]
# real_laser = preds["true_params"]
# real_emissivity = preds["true_emiss"]
# predicted_emissivity = preds["pred_emiss"][1]
# save_integral_emiss_point(predicted_emissivity, real_emissivity, "test.txt")