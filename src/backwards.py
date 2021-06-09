import torch
import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

MongoClient_uri = "mongodb://propopt_ro:2vsz634dwrwwsq@mongodb07.nersc.gov/propopt"
client = pymongo.MongoClient(MongoClient_uri)
db = client.propopt

input_df = pd.DataFrame(db.laser_samples.find(projection={"emissivity_spectrum": 0, 
            'emissivity_averaged_over_wavelength' : 0,
			'emissivity_averaged_over_frequency' : 0,
			'emissivity_spectrum' : 0,
			'laser_hardware_model' : 0,
			'laser_polarization' : 0,
			'laser_power_W' : 0,
			'laser_repetition_rate_kHz' : 1,
			'laser_scanning_line_spacing_x_dir_micron' : 0,
			'laser_scanning_line_spacing_y_dir_micron' : 1,
			'laser_scanning_speed_x_dir_mm_per_s' : 1,
			'laser_scanning_speed_y_dir_mm_per_s' : 0,
			'laser_steering_equipment': 0,
			'laser_wavelength_nm' : 0,
			"substrate_details" : 0,
			'substrate_label' : 0,
			'substrate_material' : 0}))

emiss_df = pd.DataFrame(db.laser_samples.find(projection={"emissivity_spectrum": 0, 
            'emissivity_averaged_over_wavelength' : 0,
			'emissivity_averaged_over_frequency' : 0,
			'emissivity_spectrum' : 1,
			'laser_hardware_model' : 0,
			'laser_polarization' : 0,
			'laser_power_W' : 0,
			'laser_repetition_rate_kHz' : 0,
			'laser_scanning_line_spacing_x_dir_micron' : 0,
			'laser_scanning_line_spacing_y_dir_micron' : 0,
			'laser_scanning_speed_x_dir_mm_per_s' : 0,
			'laser_scanning_speed_y_dir_mm_per_s' : 0,
			'laser_steering_equipment': 0,
			'laser_wavelength_nm' : 0,
			"substrate_details" : 0,
			'substrate_label' : 0,
			'substrate_material' : 0}))


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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        return x

net = Net()

params = list(net.parameters())
print(len(params))
print(params[0].size()) 

input_tensor = torch.tensor(np.transpose(input_df.values))

emiss_df.columns = ['emisses']
emiss_tensor = torch.tensor(input_df['emisses'].values)

input = zip(input_tensor, emiss_tensor)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
