from pathlib import Path
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Literal, Mapping, Optional
from math import floor
import random
#from sklearn.neighbors import NearestNeighbors
#from sklearn.metrics import mean_squared_error
#from torch._C import UnionType
#from scipy import stat
import utils

def unnormalize(normed, min, max):
    return normed*(max-min)+min


#importing the data
wavelength = torch.load(Path("wavelength.pt"))
real_laser= torch.load(Path("params_true_back.pt"))
real_emissivity, preds = torch.load(Path("emiss_true_back.pt")), torch.load(Path("preds.pt"))
preds = preds[0]
predicted_emissivity = preds["pred_emiss"]

#laser indexed [vae out of 50][wavelength out of 821][params, 14]
predicted_laser = preds["params"]

arbitrary_index = 20

val_real_laser = real_laser

#splits = split(len(predicted_laser))
temp_predicted_laser = predicted_laser

val_real_emissivity = real_emissivity

#splits = split(len(predicted_emissivity))
val_predicted_emissivity = predicted_emissivity

#format the real validation set
watt1 = real_laser.T[4:].T.cpu()
watt2 = np.where(watt1 == 1)
watt = (watt2[1] + 2)/10
watt = np.reshape(np.array(watt),(len(watt),1))

speed = real_laser.T[:1].T.cpu()
speed = np.array(speed)

spacing = real_laser.T[1:2].T.cpu()
spacing = np.array(spacing)

ef = real_laser.T[2:3].T.cpu()
ef = np.array(ef.detach())

ew = real_laser.T[3:4].T.cpu()
ew = np.array(ew.detach())


real = []
real.append(watt)
real.append(speed)
real.append(spacing)
real = np.array(real)
watt_size = np.size(watt)
for i in range(len(real)):
    real[i] = [float(entry) for entry in real[i][0:watt_size]]
real = np.stack(real)

#extend emissivity wavelength x values
extended_max = 2.5
extended_min = 0.5
granularity = 100

extension = torch.tensor([extended_max-(i+1)/granularity*(extended_max-extended_min) for i in range(granularity)])

extended_wave = torch.cat((wavelength[0][115:935], extension))

extended_wave = np.flip(np.array(extended_wave))

predvsideal = False
if predvsideal == True:
    for p in range(4):
        plt.figure(p)
        wavelen_list = [4.605, 6.115, 3.5, 2.5]
        wavelen_index = [567, 706, 389, 100][p]
        wavelen_cutoff = wavelen_list[p]
        plt.title('Predicted Emissivity vs Ideal TPV Emitter')
        plt.show()

        temp = 1400 
        plot_index = 0
        #for wavelen_index in range(200,700,5):
        plot_index += 1
        planck = [float(utils.planck(wavelen, temp)) for wavelen in extended_wave]

        planck_max = max(planck)
        planck = [wave/planck_max for wave in planck]

        new_score = 0
        
        wavelen_cutoff = extended_wave[wavelen_index]

        #format the predicted params
        for i in range(50):
            
            
            new_laser = predicted_laser[i][wavelen_index]
            watt1 = new_laser[4:]
            watt2 = np.where(watt1 == 1)
            watt = (watt2[0] + 2)/10
            watt = np.reshape(np.array(watt),(len(watt),1))

            speed = new_laser.T[:1].T.cpu()
            speed = np.array(speed)

            spacing = new_laser.T[1:2].T.cpu()
            spacing = np.array(spacing)

            ef = new_laser.T[2:3].T.cpu()
            ef = np.array(ef.detach())

            ew = new_laser.T[3:4].T.cpu()
            ew = np.array(ew.detach())


            pred_laser = []
            pred_laser.append(watt)
            pred_laser.append(speed)
            pred_laser.append(spacing)
            watt_size = np.size(watt)
            for k in range(len(pred_laser)):
                pred_laser[k] = [float(entry) for entry in pred_laser[k][0:watt_size]]
            pred_laser = np.stack(pred_laser)


            print(pred_laser[0])
            #minspeed = 10, maxspeed = 700
            print(unnormalize(pred_laser[1], min = 10, max = 700)) 
            #min 1 max 42
            print(unnormalize(pred_laser[2], min = 1, max = 42))
            old_emiss = val_predicted_emissivity[i][wavelen_index]
            first_emiss = float(old_emiss[0])
            new_emiss = torch.cat((torch.tensor([first_emiss for j in range(100)]), old_emiss))

            plt.plot(extended_wave[0:919], new_emiss[0:919], c= 'blue', alpha = 0.2, linewidth = 1.0)
            # if i == 3:    
            #     new_score = utils.planck_emiss_prod(extended_wave, new_emiss, wavelen_cutoff, 1400)


            plt.xlabel('Wavelength (um)', fontsize = 16)
            plt.ylabel('Emissivity', fontsize = 16)

        plt.plot(extended_wave[0:919], new_emiss[0:919], c= 'blue', alpha = 0.2, linewidth = 1.0, label = 'Predicted Emissivities')

        # rand_index = random.randint(1,399)
        # old_emiss = np.flip(np.array(real_emissivity[rand_index].cpu()))
        # first_emiss = float(old_emiss[0])
        # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
        # max_FoMM = utils.planck_emiss_prod(extended_wave, new_emiss, wavelen_cutoff, 1400)

        # plt.plot(extended_wave[0:919], new_emiss[0:919], c= 'red', alpha = 0.1, label = 'Random training data')

        # #make random emiss curves from real validation
        # max_index = 0
        # for i in range(20):
        #     rand_index = random.randint(1,399)
        #     old_emiss = np.flip(np.array(real_emissivity[rand_index].cpu()))
        #     first_emiss = float(old_emiss[0])
        #     new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
        #     new_FoMM = utils.planck_emiss_prod(extended_wave, new_emiss, wavelen_cutoff, 1400)
        #     if new_FoMM > max_FoMM:
        #         max_FoMM = new_FoMM
        #         max_index = i
        #     plt.plot(extended_wave[0:919], new_emiss[0:919], c= 'red', alpha = 0.1, linewidth = 2.0)


        # #Best Random
        # old_emiss = np.flip(np.array(real_emissivity[max_index].cpu()))
        # first_emiss = float(old_emiss[0])
        # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
        # plt.plot(extended_wave[0:919], new_emiss[0:919], c= 'maroon', alpha = 1, linewidth = 3.0, label = f'Best random, FoM {round(float(max_FoMM),2)}')


        #Best Predicted (fixed wavelen and index)
        old_emiss = val_predicted_emissivity[3][wavelen_index]
        first_emiss = float(old_emiss[0])
        new_emiss = torch.cat((torch.tensor([first_emiss for j in range(100)]), old_emiss))

        #Pristine stainless
        with open(f"src/pristine_stainless_x.txt", 'r') as f:
            lines = f.readlines()

        pristine_x_old = [float(line[0:6]) for line in lines[228:1866]]

        pristine_x = pristine_x_old + [extended_max-(i+1)/granularity*(extended_max-extended_min) for i in range(granularity)]


        with open(f"src/pristine_stainless_y.txt", 'r') as f:
            lines = f.readlines()

        pristine_y_old = [float(line[0:6]) for line in lines[228:1866]]

        first_emiss = pristine_y_old[-1]
        pristine_y = pristine_y_old + [first_emiss for i in range(granularity)]


        pristine_FoMM = utils.planck_emiss_prod(pristine_x, pristine_y, wavelen_cutoff, 1400)

        plt.plot(pristine_x, pristine_y, '--', color = 'grey', linewidth = 2.0, label = f'Plain substrate, FoM = {round(float(pristine_FoMM),2)}')

        plt.plot(extended_wave[0:919], utils.step_at_n(extended_wave, wavelen_cutoff)[0:919], c= 'black', label = f'Ideal target emissivity', linewidth = 2.0)

        #plt.plot(extended_wave[0:919], new_emiss[0:919], c= 'limegreen', alpha = 1, label = f'Best generated design, FoM {round(float(new_score),2)}', linewidth = 3.0)

        #plt.plot(extended_wave[0:919], planck[0:919], c= 'red', label = f'planck temp {temp} K', linewidth = 2.0)

        leg = plt.legend(loc = 'upper right', prop={'size': 8})

        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
            legobj.set_alpha(1)

        leg.legendHandles[0].set_alpha(0.4)

        plt.savefig('vs_training_pristine_'+str(wavelen_index)+'.png', dpi = 300)

Laser_E_P_list = []
Laser_E_M_list = []
Laser_P_M_list = []
Emiss_E_P_list = []
Emiss_E_M_list = []
Emiss_P_M_list = []


#randomly sample from real validation

    

for q in range(144):
    # rand_index = random.randint(1,399)
    # old_emiss = np.flip(np.array(real_emissivity[rand_index].cpu()))
    # first_emiss = float(old_emiss[0])
    # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))

    vae_index = random.randint(0,49)
    wavelen_index = q*5+100
    wavelen_cutoff = extended_wave[wavelen_index]
    #set whichever wavelength the step function occurs at
    
    old_emiss = utils.step_at_n(extended_wave, wavelen_cutoff)

    new_laser = predicted_laser[arbitrary_index][wavelen_index]
    #format the predicted params
    watt1 = new_laser.T[4:].T.cpu()
    watt2 = np.where(watt1 == 1)
    watt = (watt2[0] + 2)/10
    watt = np.reshape(np.array(watt),(len(watt),1))

    speed = new_laser.T[:1].T.cpu()
    speed = np.array(speed)

    spacing = new_laser.T[1:2].T.cpu()
    spacing = np.array(spacing)

    ef = new_laser.T[2:3].T.cpu()
    ef = np.array(ef.detach())

    ew = new_laser.T[3:4].T.cpu()
    ew = np.array(ew.detach())


    pred_laser = []
    pred_laser.append(watt)
    pred_laser.append(speed)
    pred_laser.append(spacing)
    watt_size = np.size(watt)
    for k in range(len(pred_laser)):
        pred_laser[k] = [float(entry) for entry in pred_laser[k][0:watt_size]]
    pred_laser = np.stack(pred_laser)

    val_hat = pred_laser

    #sort by watt, then speed, then spacing
    real_emiss_list = old_emiss
    predicted_emiss_list = val_predicted_emissivity[vae_index][wavelen_index]
    RMSE_E_P = 0
    for wavelen_i in range(819):
        RMSE_E_P += (real_emiss_list[wavelen_i]-predicted_emiss_list[wavelen_i])**2
    Emiss_E_P_list.append(RMSE_E_P.item()/820)

    diff_expected_predicted = extended_wave[wavelen_index]
    Laser_E_P_list.append(diff_expected_predicted)

plt.figure(8)
plt.plot(Laser_E_P_list, Emiss_E_P_list, 'o')
plt.title("Laser Params vs Emiss")
plt.xlabel("Wavelength Cutoff")
plt.ylabel("Emissivity Residuals")
# plt.annotate("r-squared = {:.3f}".format(r_value), (0, 1))
plt.show()
plt.savefig('tempEPgraph.png')
# plt.plot(x1,y1,'-r')



    # max_FoMM = 1.5
    # cutoff_adjusted_max = 1

    # for wavelen_index in range(520,819): #starts at 5.9um ish
    #     wavelen_cutoff = extended_wave[wavelen_index] # wavelen_index 0 indexes 2.5 um
    #     for i in range(50):

    #         #extend wavelen to 0.5
            
    #         old_emiss = val_predicted_emissivity[i][wavelen_index]
    #         first_emiss = float(old_emiss[0])
    #         new_emiss = torch.cat((torch.tensor([first_emiss for i in range(100)]), old_emiss))

    #         new_score = utils.planck_emiss_prod(extended_wave, new_emiss, wavelen_cutoff, 1400)

    #         # cutoff_adjusted_score = new_score*((11.5-wavelen_cutoff)/wavelen_cutoff)
    #         # if cutoff_adjusted_score > cutoff_adjusted_max:
    #         #     cutoff_adjusted_max = cutoff_adjusted_score
    #         #     print("NEW BEST")
    #         #     print(cutoff_adjusted_score)
    #         #     print("wavelen cutoff")
    #         #     print(wavelen_cutoff)
    #         #     print("wavelen index")
    #         #     print(wavelen_index)
    #         #     print("index")
    #         #     print(i)
    #         #     print("new score")
    #         #     print(new_score)
                
    #         if new_score > max_FoMM:
    #             max_FoMM = new_score
    #             # print("Not best")
    #             # print(cutoff_adjusted_score)
    #             print("wavelen cutoff")
    #             print(wavelen_cutoff)
    #             print("wavelen index")
    #             print(wavelen_index)
    #             print("index")
    #             print(i)
    #             print("new score")
    #             print(new_score)
                

