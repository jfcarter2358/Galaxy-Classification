import os
from shutil import copyfile
from random import shuffle
import pandas as pd

files = [f for f in os.listdir('./reduced')]
x = list(range(0, len(files)))
shuffle(x)

split = int(0.8 * len(x))
train, test = x[:split], x[split:]
df = pd.read_csv('./classification_data/labels.csv')

for t in train:
    i = files[t][:files[t].find('.')]
    v = df[df['GalaxyID'] == int(i)]['value'].tolist()[0]
    copyfile('./reduced/' + files[t], './classification_data/training/{}.{}'.format(v,files[t]))
for t in test:
    i = files[t][:files[t].find('.')]
    v = df[df['GalaxyID'] == int(i)]['value'].tolist()[0]
    copyfile('./reduced/' + files[t], './classification_data/testing/{}.{}'.format(v,files[t]))
