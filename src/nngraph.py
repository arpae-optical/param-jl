import math
import random
from math import floor
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import mean_squared_error
# from torch._C import UnionType
# from scipy import stat
import utils


def unnormalize(normed, min, max):
    return normed * (max - min) + min


def emiss_error_graph(predicted_emissivity, real_emissivity):
    # Emiss residuals
    RMSE_total = 0
    MAPE_total = 0
    wavelength = torch.load(Path("wavelength.pt"))
    wavelength = np.flip(np.array(wavelength.cpu())[0])
    fig = plt.figure()
    best_run_index = 0
    best_RMSE = 1
    worst_RMSE = 0
    RMSE_list = []
    worst_run_index = 0
    for i_run_index in range(50):

        # Emiss residuals

        current_list = predicted_emissivity[i_run_index]

        real_emiss_list = real_emissivity[i_run_index]

        # old_emiss = predicted_emissivity[i_run_index][0:820]
        # first_emiss = float(old_emiss[0])
        # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
        
        MSE_E_P = 0
        for wavelen_i in range(820):
            MSE_E_P += (real_emiss_list[wavelen_i] - current_list[wavelen_i]) ** 2
        RMSE_E_P = float(MSE_E_P / 820) ** (0.5)
        RMSE_total += RMSE_E_P/50

        RMSE_list.append(RMSE_E_P)

        if RMSE_E_P < best_RMSE:
            best_RMSE = RMSE_E_P
            best_run_index = i_run_index
        
        if RMSE_E_P > worst_RMSE:
            worst_RMSE = RMSE_E_P
            worst_run_index = i_run_index

        MAPE = 0
        for wavelen_i in range(820):
            MAPE += abs(real_emiss_list[wavelen_i] - current_list[wavelen_i])
        MAPE = float(MAPE / 820)
        MAPE_total += MAPE/50

    RMSE_residuals = [abs(RMSE_total - r) for r in RMSE_list]
    average_run_index = min(range(len(RMSE_residuals)), key=RMSE_residuals.__getitem__)

    old_emiss = predicted_emissivity[1]
    # first_emiss = float(old_emiss[0])
    # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
    plt.scatter(
        wavelength[0:820],
        [0 for n in range(820)],
        s=[0.001 for n in range(820)],
        label="Point density for reference",
    )

    

    # plt.plot(
    #     wavelength[0:820],
    #     old_emiss[0:820],
    #     c="blue",
    #     alpha=0.1,
    #     linewidth=2.0,
    #     label=f"Predicted Emiss, average RMSE {round(RMSE_total/50,5)}, MAPE {round(MAPE_total/50,5)}",
    # )

    # new_emiss = real_emissivity[1]
    # plt.plot(
    #     wavelength[0:820],b
    #     new_emiss[0:820],
    #     c="black",
    #     alpha=1,
    #     linewidth=2.0,
    #     label="Real Emissivity",
    # )

    # leg = plt.legend(loc="upper right", prop={"size": 8})

    # leg.legendHandles[0].set_alpha(1)
    # plt.xlabel("Wavelength")
    # plt.ylabel("Emissivity")

    best_RMSE_pred = predicted_emissivity[best_run_index][0:820]

    best_RMSE_real = real_emissivity[best_run_index][0:820]

    worst_RMSE_pred = predicted_emissivity[worst_run_index][0:820]

    worst_RMSE_real = real_emissivity[worst_run_index][0:820]

    average_RMSE_pred = predicted_emissivity[average_run_index][0:820]

    average_RMSE_real = real_emissivity[average_run_index][0:820]

    return([best_RMSE_pred, best_RMSE_real, worst_RMSE_pred, worst_RMSE_real, average_RMSE_pred, average_RMSE_real, wavelength, RMSE_total])


def graph(residualsflag, predsvstrueflag, index_str="default", target_str="0"):
    # importing the data
    wavelength = torch.load(Path("wavelength.pt"))
    wavelength = np.flip(np.array(wavelength.cpu())[0])
    preds = torch.load(
        Path(f"{target_str}")
    )  # this'll just be str() if I end up not needing it
    preds = preds[0]
    real_laser = preds["true_params"]
    real_emissivity = preds["true_emiss"]
    predicted_emissivity = preds["pred_emiss"]

    # laser indexed [vae out of 50][wavelength out of 821][params, 14]
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
        for p in range(0, 400):

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

                temp_real_laser = real_laser[i_run_index]

                temp_predicted_laser = predicted_laser[arbitrary_vae][i_run_index]

                # Emiss residuals
                j = arbitrary_vae
                current_list = predicted_emiss_list[j][i_run_index][0:820]

                MSE_E_P = 0
                for wavelen_i in range(820):
                    MSE_E_P += (
                        real_emiss_list[wavelen_i] - current_list[wavelen_i]
                    ) ** 2
                MSE_E_P = MSE_E_P / 820

                # Laser Param Residuals
                watt1 = temp_real_laser.T[2:].T.cpu()
                watt2 = np.where(watt1 == 1)
                watt = (watt2[0] + 2) / 10

                speed = temp_real_laser.T[:1].T.cpu()
                speed = np.array(speed)

                spacing = temp_real_laser.T[1:2].T.cpu()
                spacing = np.array(spacing)

                real = []
                real.append(watt)
                real.append(speed)
                real.append(spacing)
                real = np.array(real)
                watt_size = np.size(watt)
                for i in range(len(real)):
                    real[i] = [float(entry) for entry in real[i][0:watt_size]]
                real = np.stack(real)

                watt1 = temp_predicted_laser.T[2:].T.cpu()
                watt2 = np.where(watt1 == 1)
                watt = (watt2[0] + 2) / 10

                speed = temp_predicted_laser.T[:1].T.cpu()
                speed = np.array(speed)

                spacing = temp_predicted_laser.T[1:2].T.cpu()
                spacing = np.array(spacing)

                predicted = []
                predicted.append(watt)
                predicted.append(speed)
                predicted.append(spacing)
                predicted = np.array(predicted)
                watt_size = np.size(watt)
                for i in range(len(predicted)):
                    predicted[i] = [float(entry) for entry in predicted[i][0:watt_size]]
                predicted = np.stack(predicted)
                MSE_laser = (
                    (predicted[0] - real[0]) ** 2
                    + (predicted[1] - real[1]) ** 2
                    + (predicted[2] - real[2]) ** 2
                ) / 3

                watt = float(watt)
                speed = float(unnormalize(speed, min=10, max=700))
                spacing = float(unnormalize(spacing, min=1, max=42))

                file = open(f"{index_str}_watt_speed_spacing.csv", "a")
                file.write(f"{watt}, {speed}, {spacing} \n")
                file.close()

                MSE_E_P = float(MSE_E_P)
                MSE_laser = float(MSE_laser)
                bucket_cutoff = 0.03
                if MSE_E_P >= bucket_cutoff:
                    buckets[floor(MSE_laser / 0.1)] += 1
                    bucket_totals[floor(MSE_laser / 0.1)] += 1
                if MSE_E_P < bucket_cutoff:
                    bucket_totals[floor(MSE_laser / 0.1)] += 1

                params_list.append(MSE_laser)

                MSE_list.append(MSE_E_P)

        print("Emiss Std")
        print(np.std(MSE_list))
        print(np.mean(MSE_list))

        print("Param Std")
        print(np.std(params_list))
        print(np.mean(params_list))
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        c = np.linspace(0, 10, len(params_list))
        cmap = cm.jet
        s = [4 for n in range(len(params_list))]
        ax.scatter(params_list, MSE_list, alpha=0.1, s=s, c=c, cmap=cmap)
        bucket_x = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        bucket_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(10):
            print(i)
            print("bucket fill")
            print(buckets[i])
            print("out of")
            print(bucket_totals[i])
            if bucket_totals[i] == 0:
                bucket_y[i] = 0
            else:
                bucket_y[i] = buckets[i] / bucket_totals[i] * 5
        bucket_y = bucket_y[0:5]
        bucket_x = bucket_x[0:5]
        plt.bar(bucket_x, bucket_y, align="center", alpha=0.3, width=0.1)
        plt.title("Laser Params vs Emiss")
        plt.xlabel("Laser Parameter Residuals")
        plt.ylabel("Emissivity Residuals")
        # plt.annotate("r-squared = {:.3f}".format(r_value), (0, 1))
        plt.savefig(f"{index_str}_graph_buckets.png")

    Laser_E_P_list = []
    Laser_E_M_list = []
    Laser_P_M_list = []
    Emiss_E_P_list = []
    Emiss_E_M_list = []
    Emiss_P_M_list = []

    # randomly sample from real validation

    if predsvstrueflag == True:
        for i_run_index in range(0, 50, 1):

            plt.figure(100 + i_run_index)
            RMSE_total = 0
            MAPE_total = 0
            for arbitrary_vae in range(50):
                print("vae: " + str(arbitrary_vae))

                # Emiss residuals

                current_list = predicted_emissivity[arbitrary_vae][i_run_index][0:820]

                real_emiss_list = real_emissivity[i_run_index]

                old_emiss = predicted_emissivity[arbitrary_vae][i_run_index]
                # first_emiss = float(old_emiss[0])
                # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
                plt.plot(
                    wavelength[0:820],
                    old_emiss[0:820],
                    c="blue",
                    alpha=0.1,
                    linewidth=2.0,
                )
                MSE_E_P = 0
                for wavelen_i in range(820):
                    MSE_E_P += (
                        real_emiss_list[wavelen_i] - current_list[wavelen_i]
                    ) ** 2
                RMSE_E_P = float(MSE_E_P / 820) ** (0.5)
                RMSE_total += RMSE_E_P

                MAPE = 0
                for wavelen_i in range(820):
                    MAPE += abs(real_emiss_list[wavelen_i] - current_list[wavelen_i])
                MAPE = float(MAPE / 820)
                MAPE_total += MAPE

            old_emiss = predicted_emissivity[49][i_run_index]
            # first_emiss = float(old_emiss[0])
            # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
            plt.scatter(
                wavelength[0:820],
                [0 for n in range(820)],
                s=[0.001 for n in range(820)],
                label="Point density for reference",
            )
            plt.plot(
                wavelength[0:820],
                old_emiss[0:820],
                c="blue",
                alpha=0.1,
                linewidth=2.0,
                label=f"Predicted Emiss, average RMSE {round(RMSE_total/50,5)}, MAPE {round(MAPE_total/50,5)}",
            )

            new_emiss = real_emissivity[i_run_index]
            plt.plot(
                wavelength[0:820],
                new_emiss[0:820],
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
# print(emiss_error_graph(predicted_emissivity, real_emissivity))