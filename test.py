import wavefinder as wf
import numpy as np
import csv

path = "jack_data/Exp videos/fluid 1"

reach = [448,554]
reach[1] += 1
data = {}

for i in range(reach[1]-reach[0]):
    name = f'/MVI_0{reach[0]+i}.MP4'
    currpath = path + name
    data[name] = wf.retina(currpath, livePlay=False, singular=False, display=False, save=True, nc_points=1000)
    print(f'\n video {i+1}/{reach[1]-reach[0]} scanned')

wf.clear_bar()

print('forming csv')
maxlen = max(data[d].shape[1] for d in data)
net = np.zeros((maxlen, len(data)))
for id, (name, arr) in enumerate(data.items()):
    net[:len(arr[1]), id] = arr[1]

headers = list(data.keys())

with open('fluid1.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(headers)
    w.writerows(net)

print('\rfile saved')