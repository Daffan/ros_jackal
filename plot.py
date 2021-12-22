import tensorflow as tf
import matplotlib.pyplot as plt

path = "logging/motion_control_continuous_laser-v0/OLD/2021_08_25_00_37/events.out.tfevents.1629869868.eldar-1"
path2 = "logging/motion_control_continuous_laser-v0/MT_TD3/2021_08_25_20_18/events.out.tfevents.1629940697.eldar-1"

data = {}
for summary in tf.train.summary_iterator(path):
    try:
        x = summary.summary.value[0]
        if x.tag not in data:
            data[x.tag] = []
        data[x.tag].append(x.simple_value)
    except:
        continue

#import pdb; pdb.set_trace()

#plt.plot(data["train/Steps"], data["train/Episode_return1"], label='ret1')
plt.plot(data["train/Steps"], data["train/Episode_return2"], label='mt')

data = {}
for summary in tf.train.summary_iterator(path2):
    try:
        x = summary.summary.value[0]
        if x.tag not in data:
            data[x.tag] = []
        data[x.tag].append(x.simple_value)
    except:
        continue

plt.plot(data["train/Steps"], data["train/Episode_return2"], label='sgd')
plt.yscale('symlog')
plt.legend()

plt.savefig("mt.png")

