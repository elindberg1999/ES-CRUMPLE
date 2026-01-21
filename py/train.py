import sys, os

if len(sys.argv) < 6:
    print('Usage: python train.py <round index> <num of sim per round> <relative parm file path> <just the traj> <initial coord file name>')
    exit()

import numpy as np

import glob
import PIL
import tensorflow as tf
from tensorflow import keras
import time
from modelAE import *
from sklearn.cluster import DBSCAN
import MDAnalysis as mda
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

tf.config.optimizer.set_jit(False)
tf.config.run_functions_eagerly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

round_idx = int(sys.argv[1])
n_sim = int(sys.argv[2])
psf = sys.argv[3]
dcd_fname = sys.argv[4]
init_fname = sys.argv[5]

CM_this = []

input_size = int(np.load(f'../Simulations/data/input_size.npy'))

if round_idx > 0:
    print(f"Loading npy of previous rounds: round 0 to {round_idx-1} [WARNING: NOT USED IN CURRENT CLUSTERING]")
    CM_prev = np.load(f'../Simulations/data/rounds_0_to_{round_idx-1}.npy')


for i in range(n_sim):
    print(f"Loading npy of this round: idx {i}")
    try:
        CM_this.append(np.load(f'../Simulations/data/{round_idx}_{i}.npy'))
    except Exception as e:
        print(e)

CM_this = np.concatenate(CM_this, axis=0)
CM_size = len(CM_this)

if round_idx > 0:
    CM_all = np.concatenate([CM_prev, CM_this], axis=0)
else:
    CM_all = CM_this

print(f"Saving npys of round 0 to {round_idx}")
np.save(f'../Simulations/data/rounds_0_to_{round_idx}.npy', CM_all)

np.random.shuffle(CM_all)

n_res = len(CM_all[0])
n_frames = len(CM_all)

if n_frames > 480000:

    CM_all = CM_all[:96000]

print(f'There are {n_res} dihedrals and {n_frames} frames')

CM_tuple = [tuple(x) for x in list(CM_all[:,:])]

#CM_category = [CM_category_map[x] for x in CM_tuple] # This is a map of frame -> combination index

print("Preprocess data")
frames, data =CM_all[:,:3].astype('int'), pd.DataFrame(CM_all[:,3:])

train_data = data.sample(frac=0.8, random_state=0)
valid_data = data.drop(train_data.index)

batch_size = min(2048, max(int(len(train_data) / 8) // 16 * 16, 1))

elbo_history = []

epochs = int(max(np.floor(2000/((round_idx)+1)), 500))

print(f"Number of epochs for this round: {epochs}")

model.build(input_shape=(None, input_size))

if round_idx > 0:
    latest = f"../Simulations/saved_models/round_{round_idx-1}/checkpoint.weights.h5"
    try:
        model.load_weights(latest)
        print(f'Loaded pretrained model from rounds 0 to {round_idx-1}')
    except Exception as e:
        print("Could not load previous weights:", e)
else:
    print("Round 0: Start fresh model")

os.makedirs(f'../Simulations/saved_models/round_{round_idx}', exist_ok=True)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=f"../Simulations/saved_models/round_{round_idx}/checkpoint.weights.h5",
    save_weights_only=True,
    save_best_only=True
)

model.compile(
    optimizer='adam',
    loss='mse',       # or another loss appropriate for your task
    metrics=['mae']
)

model.fit(train_data, train_data,
                epochs=epochs,
                shuffle=True,
                validation_data=(valid_data,valid_data),
                batch_size=batch_size,
                callbacks=[checkpoint],
                verbose=2)

frameX = round_idx * CM_size

if round_idx > 4:

    frameX = 0

data2 = data.iloc[frameX:,:]
frames2 = frames[frameX:]

datanp = data2.to_numpy()

print("Encode all data ...")
CM_embed = model.encode(datanp)
#CM_embed = model.encode(CM_list)
print(CM_embed.shape)

np.savetxt(f"latent_embeddings/{round_idx}_data.csv",CM_embed,delimiter=",")

print("Running DBSCAN in latent space ...")
eps_init = 0.3
if round_idx > 0:
    try:
        with open('../Simulations/eps','r') as f:
            eps_init = float(f.read().strip())
    except Exception as e:
        print(e)
if eps_init < 0.25:
    eps_choices = np.linspace(0.45, 0.05, num = 19)
else:
    eps_choices = np.linspace(eps_init + 0.2, eps_init - 0.2, num = 19)

min_samples_choice = latent_dim

#eps_choices = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5][::-1]
num_outliers = []
cls_collection = []
t0 = time.time()
for eps in eps_choices:
    cls = DBSCAN(eps=eps, n_jobs=-1, min_samples=min_samples_choice, algorithm='kd_tree')
    cls.fit(CM_embed)
    print("Suggesting outliers in latent space ...")
    sum_outliers = np.sum(cls.labels_ == -1)
    num_outliers.append(sum_outliers)
    cls_collection.append(cls)
    t1 = time.time()
    print(f'There are {sum_outliers} outliers for eps = {eps} and min_samples = {min_samples_choice} ({(t1-t0)*1000:.1f} ms)')
    if sum_outliers >= n_sim:
        continue

# Select best DBSCAN model ... best is exactly n_sim outliers, 
# then slightly more than n_sim outliers,
# then a lot more, then a little less, then none
outlier_rank = np.array(num_outliers)
outlier_rank[outlier_rank < n_sim] = outlier_rank.max() + n_sim + 1 - outlier_rank[outlier_rank < n_sim]
cls_sel = np.argmin(outlier_rank)
print(f'Using eps = {eps_choices[cls_sel]}, and there are {num_outliers[cls_sel]} outliers')
cls = cls_collection[cls_sel]
t1 = time.time()
print(f'Determining best eps value took {(t1-t0)*1000:.1f} ms')

#category_to_cls_label = {}
#for idx, x in enumerate(cls.labels_):
#    category_to_cls_label[idx] = x

# Put back assignment
#CM_label = np.array([category_to_cls_label[x] for x in CM_category])
outliers = frames2[cls.labels_ == -1]

np.savetxt(f"labels/labels{round_idx}.csv",cls.labels_,delimiter=",")
np.savetxt(f"frames/frames{round_idx}.csv",frames2,delimiter=",")

print(outliers)
print(f'There are {len(outliers)} outliers')

if len(outliers) < n_sim:
    extra_select = frames2[np.random.choice(np.array(np.arange(len(data)))[np.nonzero(cls.labels_ > -1)], n_sim - len(outliers), replace=False)]
    if len(outliers) == 0:
        select = extra_select
    else:
        select = np.vstack((outliers, extra_select))
elif len(outliers) > n_sim:
    select = outliers[np.random.choice(len(outliers), n_sim, replace=False)]
else: # len outliers match n_sim
    select = outliers

print(select)

print("Finding and outputting specific frames ...")

for idx, sel in enumerate(select):
    os.makedirs(f'../Simulations/{round_idx+1}/{idx}', exist_ok=True)
    U = mda.Universe(psf, f'../Simulations/{sel[0]}/{sel[1]}/{dcd_fname}',format="NCDF")
    U.trajectory[sel[2]]
    Uall = U.select_atoms('all')
    with mda.coordinates.TRJ.NCDFWriter(f'../Simulations/{round_idx+1}/{idx}/ncdf_{init_fname}', Uall.atoms.n_atoms) as W:
        W.write(Uall)

with open('../Simulations/eps', 'w') as f:
    f.write(f'{eps_choices[cls_sel]:.2f}')
