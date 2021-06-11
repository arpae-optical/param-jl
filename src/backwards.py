import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.optim as optim

MongoClient_uri = "mongodb://propopt_ro:2vsz634dwrwwsq@mongodb07.nersc.gov/propopt"
client = pymongo.MongoClient(MongoClient_uri)
db = client.propopt

#input_df = pd.DataFrame(db.laser_samples.find()).to_numpy()


#emiss_df = pd.DataFrame(db.laser_samples.find(projection={
#			'emissivity_spectrum' : 1})).to_numpy()


emiss_list = []
x_speed_list = []
y_spacing_list = []
frequency_list = []

for entry in db.laser_samples.find():
    EmissPlot = []
    emiss = entry["emissivity_spectrum"]
    
    for ex in emiss:
        EmissPlot.append(ex["normal_emissivity"]) #so that I can check that there's 935 emisses
    
    size_flag = True

    if len(EmissPlot) != (935): #checks for 935 emisses
        size_flag = False
    
    if size_flag == True: #pushes the other 3 parameters
        x_speed_list.append(entry["laser_scanning_speed_x_dir_mm_per_s"])
        y_spacing_list.append(entry["laser_scanning_line_spacing_y_dir_micron"])
        frequency_list.append(entry["laser_repetition_rate_kHz"])
        
        emiss_list.append(EmissPlot) #pushes the list
input_tensor = []
label_tensor = []

for i in range(len(emiss_list)):
    label_tensor.append([
                x_speed_list[i],
                y_spacing_list[i],
                float(frequency_list[i])
            ]
            )
    input_tensor.append(
            emiss_list[i]
        )

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.fc1 = nn.Linear(935, 32)  # 5*5 from image dimension
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return x

net = Net()

params = list(net.parameters())

input = zip(torch.FloatTensor(input_tensor), torch.FloatTensor(label_tensor))

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

loss_list = []

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(input, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            loss_list.append(running_loss)
            running_loss = 0.0

x_points = []
for i in range(len(loss_list)):
    x_points.append(i)
plt.plot(x_points, loss_list)
plt.savefig('test_plot3.png')
print('Finished Training')