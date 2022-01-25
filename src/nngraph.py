from pathlib import Path
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Literal, Mapping, Optional
from math import floor
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
#from torch._C import UnionType
from scipy import stats


Stage = Literal["train", "val", "test"]
def split(n: int, splits: Optional[Mapping[Stage, float]] = None) -> Dict[Stage, range]:
    """
    n: length of dataset
    splits: map where values should sum to 1 like in `{"train": 0.8, "val": 0.1, "test": 0.1}`
    """
    if splits is None:
        splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    return {
        "train": range(0, floor(n * splits["train"])),
        "val": range(
            floor(n * splits["train"]),
            floor(n * splits["train"]) + floor(n * splits["val"]),
        ),
        "test": range(floor(n * splits["train"]) + floor(n * splits["val"]), n),
    }

#importing the data
wavelength = torch.load(Path("wavelength.pt"))
real_laser, predicted_laser = torch.load(Path("params_true_back.pt")), torch.load(Path("param_pred.pt"))
real_emissivity, predicted_emissivity = torch.load(Path("emiss_true_back.pt")), torch.load(Path("emiss_pred.pt"))


#spliting the data
splits = split(len(real_laser))

train_real_laser = real_laser[splits["train"].start : splits["train"].stop]
val_real_laser = real_laser[splits["val"].start : splits["val"].stop]
test_real_laser = real_laser[splits["test"].start : splits["test"].stop]

splits = split(len(predicted_laser))
train_predicted_laser = predicted_laser[splits["train"].start : splits["train"].stop]
val_predicted_laser = predicted_laser[splits["val"].start : splits["val"].stop]
test_predicted_laser = predicted_laser[splits["test"].start : splits["test"].stop]

splits = split(len(real_emissivity))
train_real_emissivity = real_emissivity[splits["train"].start : splits["train"].stop]
val_real_emissivity = real_emissivity[splits["val"].start : splits["val"].stop]
test_real_emissivity = real_emissivity[splits["test"].start : splits["test"].stop]

splits = split(len(predicted_emissivity))
train_predicted_emissivity = predicted_emissivity[splits["train"].start : splits["train"].stop]
val_predicted_emissivity = predicted_emissivity[splits["val"].start : splits["val"].stop]
test_predicted_emissivity = predicted_emissivity[splits["test"].start : splits["test"].stop]

# all the real data for Nearest Neighbors
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
real.append(ef)
real.append(ew)
real = np.array(real)
watt_size = np.size(watt)
for i in range(len(real)):
    real[i] = [np.float(entry) for entry in real[i][0:watt_size]]
real = np.stack(real)

#the predictions from the validation set
val_hat_watt1 = val_predicted_laser.T[4:].T.cpu()
val_hat_watt2 = np.where(val_hat_watt1 == 1)
val_hat_watt = (val_hat_watt2[1] + 2)/10
val_hat_watt = np.reshape(np.array(val_hat_watt),(len(val_hat_watt),1))

val_hat_speed = val_predicted_laser.T[:1].T.cpu()
val_hat_speed = np.array(val_hat_speed.detach())

val_hat_spacing = val_predicted_laser.T[1:2].T.cpu()
val_hat_spacing = np.array(val_hat_spacing.detach())

val_hat_ef = val_predicted_laser.T[2:3].T.cpu()
val_hat_ef = np.array(val_hat_ef.detach())

val_hat_ew = val_predicted_laser.T[3:4].T.cpu()
val_hat_ew = np.array(val_hat_ew.detach())

val_hat = []
val_hat.append(val_hat_watt)
val_hat.append(val_hat_speed)
val_hat.append(val_hat_spacing)
val_hat.append(val_hat_ef)
val_hat.append(val_hat_ew)
val_hat = np.array(val_hat)
watt_size = np.size(val_hat_watt)
print(watt_size)
for i in range(len(val_hat)):
    val_hat[i] = [np.float(entry) for entry in val_hat[i][0:watt_size]]
val_hat = np.stack(val_hat)


#the real data from the validation set
val_watt1 = val_real_laser.T[4:].T.cpu()
val_watt2 = np.where(val_watt1 == 1)
val_watt = (val_watt2[1] + 2)/10
val_watt = np.reshape(np.array(val_watt),(len(val_watt),1))

val_speed = val_real_laser.T[:1].T.cpu()
val_speed = np.array(val_speed)

val_spacing = val_real_laser.T[1:2].T.cpu()
val_spacing = np.array(val_spacing)

val_ef = val_real_laser.T[2:3].T.cpu()
val_ef = np.array(val_ef.detach())

val_ew = val_real_laser.T[3:4].T.cpu()
val_ew = np.array(val_ew.detach())


val = []
val.append(val_watt)
val.append(val_speed)
val.append(val_spacing)
val.append(val_ef)
val.append(val_ew)
val = np.array(val)


watt_size = np.size(val_watt)
print(watt_size)
for i in range(len(val)):
    val[i] = [np.float(entry) for entry in val[i][0:watt_size]]
val = np.stack(val)



#make and fit the nearest neighbors
#neigh = NearestNeighbors(n_neighbors=1)
#neigh.fit(real.T)

#get random indexes and the corresponding indexes of the nearest neighbors
n = []
#m = []
for i in range(23):
    n.append(i)
    #m.append(neigh.kneighbors(X=val_hat.T[n[i]].reshape(1, -1), n_neighbors=1, return_distance=False))

def unnormalize(normed, min, max):
    return normed*(max-min)+min


sqrt_data_size = math.ceil((len(real_laser)/10)**0.5)


plt.suptitle('Predicted vs Expected vs Manufactured Laser Params: Random Validation', fontsize = 100)
plt.tight_layout(pad = 10)
plt.subplots_adjust(top=(1-0.5/sqrt_data_size))
plt.show()


#graph the laser params
fig = plt.figure(figsize = (sqrt_data_size*10,sqrt_data_size*10))
for i in range(22):
    ax = fig.add_subplot(sqrt_data_size, sqrt_data_size, i+1, projection = '3d')
    #sort by watt, then speed, then spacing
    print("watt")
    print(val[0][n[i]])
    print("speed")
    #minspeed = 10, maxspeed = 700
    print(unnormalize(val[1][n[i]], min = 10, max = 700))
    print("spacing")
    #min 1 max 42
    print(unnormalize(val[2][n[i]], min = 1, max = 42))
    ax.scatter(val[1][n[i]], val[0][n[i]], val[2][n[i]],s =200, c= 'black', label = 'Expected')
    ax.scatter(val_hat[1][n[i]], val_hat[0][n[i]], val_hat[2][n[i]], s =200, c = 'r', label = 'Predicted')
    


    

    ax.set_title(f'Index = {n[i]}', fontsize = 60)
    
    ax.set_xlabel('Speed', fontsize = 40, labelpad = 20)
    ax.set_ylabel('Watt', fontsize = 40, labelpad = 40)
    ax.set_zlabel('Spacing', fontsize = 40, labelpad = 20)
            
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.4)
    ax.set_zlim(0,1)

    ax.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize = 20, loc = 'upper right')

    
fig.savefig('temp_scatter.png', dpi=fig.dpi)

# graph the wavelength and emissivity
plt.suptitle('Predicted vs Expected vs Nearest Laser Params: Random Validation', fontsize = 100)
plt.tight_layout(pad = 10)
plt.subplots_adjust(top=(1-0.5/sqrt_data_size))
plt.show()
fig = plt.figure(figsize = (sqrt_data_size*10,sqrt_data_size*10))


for i in range(22):
    watt = val[0][n[i]]
    speed = unnormalize(val[1][n[i]], min = 10, max = 700)
    spacing = unnormalize(val[2][n[i]], min = 1, max = 42)
    watt_str = str(round(watt,1))
    watt_ones = watt_str[0]
    watt_tens = watt_str[2]
    speed_num = round(speed,1)
    if speed_num%1 < 0.001:
        speed_str = str(int(round(speed_num,0)))
    else:
        speed_str = str(speed_num)
    spacing_str = "blank"
    spacing_num = round(spacing,1)
    if spacing_num%1 < 0.001:
        spacing_str = str(int(round(spacing_num,0)))
    else:
        spacing_str = str(spacing_num)
    
    # with open(f"Stainless steel_Validation/{watt_ones}_{watt_tens}W/Power_{watt_ones}_{watt_tens}_W_Speed_{speed_str}_mm_s_Spacing_{spacing_str}_um.txt", 'r') as f:
    #     lines = f.readlines()

    # count = 0
    # manufactured_emiss_list = []
    # for line in lines[0:934]:
    #     manufactured_emiss_list.append(float(line[17:32]))
    ax = fig.add_subplot(sqrt_data_size, sqrt_data_size, i+1)
    
    ax.scatter(wavelength[0][115:935], val_real_emissivity.detach()[n[i]].cpu(), s =10, c= 'black', label = 'Expected')
    ax.scatter(wavelength[0][115:935], val_predicted_emissivity.detach()[n[i]].cpu(), s =10, c = 'r', label = 'Predicted')
    
    
    ax.set_title(f'Index = {n[i]}', fontsize = 60)
    ax.set_xlabel('Wavelength', fontsize = 40)
    ax.set_ylabel('Emissivity', fontsize = 40)

    ax.set_xlim(0,26)
    ax.set_ylim(0,1)

    ax.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize = 20, loc = 'lower left')
plt.suptitle('Predicted vs Expected vs Manufactured Emissivity: Random Validation', fontsize = 100)         
plt.tight_layout(pad = 10)
plt.subplots_adjust(top=(1-0.5/sqrt_data_size))
fig.savefig('temp_graph.png', dpi=fig.dpi)


    
# Laser_E_P_list = []
# Laser_E_M_list = []
# Laser_P_M_list = []
# Emiss_E_P_list = []
# Emiss_E_M_list = []
# Emiss_P_M_list = []


# for i in range(159):
    


#     watt = val[0][n[i]]
#     speed = unnormalize(val[1][n[i]], min = 10, max = 700)
#     spacing = unnormalize(val[2][n[i]], min = 1, max = 42)
#     watt_str = str(round(watt,1))
#     watt_ones = watt_str[0]
#     watt_tens = watt_str[2]
#     speed_num = round(speed,1)
#     if speed_num%1 < 0.001:
#         speed_str = str(int(round(speed_num,0)))
#     else:
#         speed_str = str(speed_num)
#     spacing_str = "blank"
#     spacing_num = round(spacing,1)
#     if spacing_num%1 < 0.001:
#         spacing_str = str(int(round(spacing_num,0)))
#     else:
#         spacing_str = str(spacing_num)

    
#     manufactured_emiss_list = []
#     print(f"Stainless steel_Validation/{watt_ones}_{watt_tens}W/Power_{watt_ones}_{watt_tens}_W_Speed_{speed_str}_mm_s_Spacing_{spacing_str}_um.txt")
#     my_file = Path(f"Stainless steel_Validation/{watt_ones}_{watt_tens}W/Power_{watt_ones}_{watt_tens}_W_Speed_{speed_str}_mm_s_Spacing_{spacing_str}_um.txt")
#     if my_file.is_file():
#         with open(f"Stainless steel_Validation/{watt_ones}_{watt_tens}W/Power_{watt_ones}_{watt_tens}_W_Speed_{speed_str}_mm_s_Spacing_{spacing_str}_um.txt", 'r') as f:
#             lines = f.readlines()
#         count = 0
#         for line in lines[0:933]:
#             manufactured_emiss_list.append(float(line[17:32]))
    
#         real_emiss_list = val_real_emissivity.detach()[n[i]].cpu()
#         predicted_emiss_list = val_predicted_emissivity.detach()[n[i]].cpu()
#         MSE_E_P = 0
#         MSE_E_M = 0
#         MSE_P_M = 0
#         for wavelen_i in range(933):
#             MSE_E_P += (real_emiss_list[wavelen_i]-predicted_emiss_list[wavelen_i])**2
#             print(i)
#             MSE_E_M += (manufactured_emiss_list[wavelen_i]-real_emiss_list[wavelen_i])**2

#             MSE_P_M += (manufactured_emiss_list[wavelen_i]-predicted_emiss_list[wavelen_i])**2


#         Emiss_E_P_list.append(MSE_E_P/933)
#         Emiss_E_M_list.append(MSE_E_M/933)
#         Emiss_P_M_list.append(MSE_P_M/933)
#         diff_expected_predicted = (abs(val[1][n[i]]-val_hat[1][n[i]])+abs(val[0][n[i]]-val_hat[0][n[i]])+abs(val[2][n[i]]-val_hat[2][n[i]]))/3
#         Laser_E_P_list.append(diff_expected_predicted)
#         diff_expected_manufactured = (abs(real[1][n[i]]-val_hat[1][n[i]])+abs(real[0][n[i]]-val_hat[0][n[i]])+abs(real[2][n[i]]-val_hat[2][n[i]]))/3
#         Laser_E_M_list.append(diff_expected_manufactured)
#         diff_predicted_manufactured = (abs(real[1][n[i]]-val[1][n[i]])+abs(real[0][n[i]]-val[0][n[i]])+abs(real[2][n[i]]-val[2][n[i]]))/3
#         Laser_P_M_list.append(diff_predicted_manufactured)


# x = np.array(Laser_E_P_list)
# y = np.array(Emiss_E_P_list)
# # gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# # mn=np.min(x)
# # mx=np.max(x)
# # x1=np.linspace(mn,mx,500)
# # y1=gradient*x1+intercept
# plt.plot(x,y,'ob')
# # plt.plot(x1,y1,'-r')
# plt.title("Laser Params vs Emiss")
# plt.xlabel("Laser Parameters Residuals")
# plt.ylabel("Emissivity Residuals")
# # plt.annotate("r-squared = {:.3f}".format(r_value), (0, 1))
# plt.show()
# plt.savefig('temp_oops_expected_predicted_graph.png')

# plt.plot(Laser_E_M_list, Emiss_E_M_list, 'o')
# plt.savefig('expected_manufactured_graph.png')
# plt.plot(Laser_P_M_list, Emiss_P_M_list, 'o')
# plt.savefig('manufactured_predicted_graph.png')
