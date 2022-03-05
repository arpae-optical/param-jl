from pathlib import Path
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
preds = torch.load(Path(f"src/preds_0_validation"))
preds = preds[0]
real_laser = preds["true_params"]
real_emissivity = preds["true_emiss"]
predicted_emissivity = preds["pred_emiss"]

#laser indexed [vae out of 50][wavelength out of 821][params, 14]
predicted_laser = preds["params"]

arbitrary_index = 20

#splits = split(len(predicted_laser))
temp_predicted_laser = predicted_laser

val_real_emissivity = real_emissivity

#splits = split(len(predicted_emissivity))
val_predicted_emissivity = predicted_emissivity





#extend emissivity wavelength x values
extended_max = 2.5
extended_min = 0.5
granularity = 100

extension = torch.tensor([extended_max-(i+1)/granularity*(extended_max-extended_min) for i in range(granularity)])

extended_wave = torch.cat((wavelength[0][115:935], extension))

original_wave = wavelength[0][115:935]

extended_wave = np.flip(np.array(extended_wave))
predvsideal = False
plt.figure(1)
buckets = [0,0,0,0,0,0,0,0,0,0] #0-0.1, 0.1-0.2, ... , 0.9-1.0
bucket_totals = [0,0,0,0,0,0,0,0,0,0]
if predvsideal == True:
    MSE_list = []
    params_list = []
    for p in range(0,400):
        print("pred index")
        print(p)
        
        #index_array = [41,109,214,284,297,302,315,378]#,431,452
        #i_run_index = index_array[p]
        i_run_index = p

        #format the predicted params

        #sort by watt, then speed, then spacing
        
        plt.title('Predicted Emissivity vs Ideal TPV Emitter')
        plt.show()
        temp = 1400 
        for arbitrary_vae in range(50):
            new_score = 0
        
            #format the predicted params
        
            real_emiss_list = real_emissivity[i_run_index]

            predicted_emiss_list = predicted_emissivity

            temp_real_laser = real_laser[i_run_index]

            temp_predicted_laser = predicted_laser[arbitrary_vae][i_run_index]

            #Emiss residuals
            j = arbitrary_vae
            current_list = predicted_emiss_list[j][i_run_index][0:820]

            MSE_E_P = 0
            for wavelen_i in range(819):
                MSE_E_P += (real_emiss_list[wavelen_i]-current_list[wavelen_i])**2
            MSE_E_P = MSE_E_P/819


            #Laser Param Residuals
            watt1 = temp_real_laser.T[2:].T.cpu()
            watt2 = np.where(watt1 == 1)
            watt = (watt2[0] + 2)/10

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
            watt = (watt2[0] + 2)/10

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
            MSE_laser = ((predicted[0]-real[0])**2+(predicted[1]-real[1])**2+(predicted[2]-real[2])**2)/3
            
            MSE_E_P = float(MSE_E_P)
            MSE_laser = float(MSE_laser)
            if MSE_E_P >= 0.1:
                buckets[floor(MSE_laser/0.1)] += 1
                bucket_totals[floor(MSE_laser/0.1)] +=1
            if MSE_E_P < 0.1:
                bucket_totals[floor(MSE_laser/0.1)] += 1

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
    ax.scatter(params_list, MSE_list, alpha = 0.1, s = s, c = c, cmap = cmap)
    bucket_x = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    bucket_y = [0,0,0,0,0,0,0,0,0,0]
    for i in range(10):
        print(i)
        print("bucket fill")
        print(buckets[i])
        print("out of")
        print(bucket_totals[i])
        if bucket_totals[i] == 0:
            bucket_y[i] = 0
        else:
            bucket_y[i] = buckets[i]/bucket_totals[i]
    plt.bar(bucket_x, bucket_y, align = "center", alpha = 0.3, width = 0.1)
    plt.title("Laser Params vs Emiss")
    plt.xlabel("Laser Parameter Residuals")
    plt.ylabel("Emissivity Residuals")
    # plt.annotate("r-squared = {:.3f}".format(r_value), (0, 1))
    plt.savefig('tempEPgraphBuckets.png')
        # BEST CODE
        # best_index = 0
        # best_MSE = 10000
        # for i in range(50):
        #     current_list = predicted_emiss_list[i][i_run_index][0:820]

        #     MSE_E_P = 0
        #     for wavelen_i in range(819):
        #         MSE_E_P += (real_emiss_list[wavelen_i]-current_list[wavelen_i])**2
        #     MSE_E_P = MSE_E_P/819
        #     if MSE_E_P < best_MSE:
        #         best_MSE = MSE_E_P
        #         best_index = i
        # average_list = predicted_emiss_list[best_index][i_run_index][0:820]

        # MSE_E_P = best_MSE

        #AVERAGE CODE
        # average_list = predicted_emiss_list[49][i_run_index][0:820]
        # for i in range(49):
        #     average_list = average_list+predicted_emiss_list[i][i_run_index][0:820]
        # average_list = average_list/50

        # MSE_E_P = 0
        # for wavelen_i in range(819):
        #     MSE_E_P += (real_emiss_list[wavelen_i]-average_list[wavelen_i])**2
        # MSE_E_P = MSE_E_P/819


        # PLOTTING STUFF TO UNCOMMENT
        #old_emiss = predicted_emiss_list
        #first_emiss = float(old_emiss[0])
        #new_emiss = torch.cat((torch.tensor([first_emiss for j in range(100)]), old_emiss))

        # new_emiss = average_list
        # #plt.plot(extended_wave[0:919], new_emiss[0:919], c= 'blue', alpha = 0.2, linewidth = 1.0)
        # # if i == 3:    
        # #     new_score = utils.planck_emiss_prod(extended_wave, new_emiss, wavelen_cutoff, 1400)


        # plt.xlabel('Wavelength (um)', fontsize = 16)
        # plt.ylabel('Emissivity', fontsize = 16)

        # plt.plot(original_wave[0:819], new_emiss[0:819], c= 'blue', alpha = 1, linewidth = 1.0, label = f'Predicted Emissivities, MSE {MSE_E_P}')

        # OLD PLOTTING STUFF, MAYBE LATER
        # real_emiss_list = torch.load(Path(f"preds_i_validation/preds_0_validation"))[0]["true_emiss"][i_run_index]
        # old_emiss = real_emiss_list
        # first_emiss = float(old_emiss[0])
        # new_emiss = torch.cat((torch.tensor([first_emiss for j in range(100)]), old_emiss))
        # plt.plot(original_wave[0:819], new_emiss[0:819], c= 'red', alpha = 1, linewidth = 2.0, label = 'Real Emissivity')


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

# OLD PLOTTING STUFF, MAYBE LATER
        # real_emiss_list = torch.load(Path(f"preds_i_validation/preds_0_validation"))[0]["true_emiss"][i_run_index]
        # old_emiss = real_emiss_list
        # first_emiss = float(old_emiss[0])
        # new_emiss = torch.cat((torch.tensor([first_emiss for j in range(100)]), old_emiss))
        # plt.plot(original_wave[0:819], new_emiss[0:819], c= 'red', alpha = 1, linewidth = 2.0, label = 'Real Emissivity')


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



        #plt.plot(extended_wave[0:919], new_emiss[0:919], c= 'limegreen', alpha = 1, label = f'Best generated design, FoM {round(float(new_score),2)}', linewidth = 3.0)

        #plt.plot(extended_wave[0:919], planck[0:919], c= 'red', label = f'planck temp {temp} K', linewidth = 2.0)

        # PLOTTING STUFF TO UNCOMMENT
        # leg = plt.legend(loc = 'upper right', prop={'size': 8})

        # for legobj in leg.legendHandles:
        #     legobj.set_linewidth(2.0)
        #     legobj.set_alpha(1)

        # leg.legendHandles[0].set_alpha(0.4)

        # plt.savefig('vs_training_best'+str(i_run_index)+'.png', dpi = 300)



Laser_E_P_list = []
Laser_E_M_list = []
Laser_P_M_list = []
Emiss_E_P_list = []
Emiss_E_M_list = []
Emiss_P_M_list = []

print("start")

#randomly sample from real validation

secondrun = True
if secondrun == True:
    for i_run_index in range(0, 400, 40):
        
        plt.figure(100+i_run_index)
        # rand_index = random.randint(1,399)
        # old_emiss = np.flip(np.array(real_emissivity[rand_index].cpu()))
        # first_emiss = float(old_emiss[0])
        # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))

        # loss = torch.load(Path("preds_i_validation/preds_{i_run_index}_validation"))[0]["pred_loss"]
        # new_laser=torch.load(Path("preds_i_validation/preds_{i_run_index}_validation"))[0]["params"][1]
        # print(len(new_laser)
        for arbitrary_vae in range(19):
            print("vae: "+str(arbitrary_vae))
            


            #Emiss residuals
            
            current_list = predicted_emissivity[arbitrary_vae][i_run_index][0:820]
            

            real_emiss_list = real_emissivity[i_run_index]
            #format the predicted params
            # watt1 = new_laser.T[4:].T.cpu()
            # watt2 = np.where(watt1 == 1)
            # watt = (watt2[0] + 2)/10
            # watt = np.reshape(np.array(watt),(len(watt),1))

            # speed = new_laser.T[:1].T.cpu()
            # speed = np.array(speed)

            # spacing = new_laser.T[1:2].T.cpu()
            # spacing = np.array(spacing)

            # ef = new_laser.T[2:3].T.cpu()
            # ef = np.array(ef.detach())

            # ew = new_laser.T[3:4].T.cpu()
            # ew = np.array(ew.detach())


            # pred_laser = []
            # pred_laser.append(watt)
            # pred_laser.append(speed)
            # pred_laser.append(spacing)
            # watt_size = np.size(watt)
            # for k in range(len(pred_laser)):
            #     pred_laser[k] = [float(entry) for entry in pred_laser[k][0:watt_size]]
            # pred_laser = np.stack(pred_laser)

            # val_hat = pred_laser

            #sort by watt, then speed, then spacing
            #make random emiss curves from real validation
            max_index = 0
        
            old_emiss = predicted_emissivity[arbitrary_vae][i_run_index]
            # first_emiss = float(old_emiss[0])
            # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
            plt.plot(extended_wave[0:820], old_emiss[0:820], c= 'blue', alpha = 0.1, linewidth = 2.0)

        old_emiss = predicted_emissivity[20][i_run_index]
        # first_emiss = float(old_emiss[0])
        # new_emiss = np.concatenate((np.array([first_emiss for j in range(100)]), old_emiss))
        plt.plot(extended_wave[0:820], old_emiss[0:820], c= 'blue', alpha = 0.1, linewidth = 2.0, label = "Predicted Emissivities")

        new_emiss = real_emissivity[i_run_index]
        plt.plot(extended_wave[0:820], new_emiss[0:820], c= 'black', alpha = 1, linewidth = 2.0, label = "Real Emissivity")

        leg = plt.legend(loc = 'upper right', prop={'size': 8})


        leg.legendHandles[0].set_alpha(0.4)

        plt.savefig('vs_training_best'+str(i_run_index)+'.png', dpi = 300)

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
                

