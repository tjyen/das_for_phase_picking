import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import h5py

from .myutils import normalize

def print_hdf5_contents(filename):
    with h5py.File(filename, 'r') as hdf_file:
        # 遞迴函數，列印 HDF5 文件的結構
        def print_group_structure(group, indent=0):
            for name, item in group.items():
                if isinstance(item, h5py.Group):
                    print(' ' * indent + f'Group: {name}/')
                    print_group_structure(item, indent + 2)
                elif isinstance(item, h5py.Dataset):
                    print(' ' * indent + f'Dataset: {name}')
        # 呼叫遞迴函數開始列印
        print_group_structure(hdf_file)

def ploting(hdf5_filename, dir_nm):
    with h5py.File(hdf5_filename, "r") as fp:
        ds = fp["data"]
        data = ds[...]
        dt = ds.attrs["dt_s"]
        dx = ds.attrs["dx_m"]
        nx, nt = data.shape
        x = np.arange(nx) * dx
        t = np.arange(nt) * dt
        print(data.shape)
        #print(data)
        print(ds.attrs.keys())
        print([ds.attrs['dt_s'], ds.attrs['dx_m'], ds.attrs['unit']])
        
    AAA = hdf5_filename.split('.')[-2]
    picks = pd.read_csv(f"results/picks_phasenet_das/{AAA}.csv")
    BBB = hdf5_filename.split('.')[-2]
    BBB = BBB.split('/')[-1]
    
    folder_path = str(dir_nm) + '_fig/fig_full_raw/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass
    
    plt.figure()
    plt.imshow(normalize(data).T, cmap="seismic", vmin=-1, vmax=1, aspect="auto", extent=[x[0], x[-1], t[-1], t[0]], interpolation="none")
    plt.xlabel("Distance (m)")
    plt.ylabel("Time (s)")
    plt.savefig(folder_path + BBB + '.png')
    #plt.close()
    
    
    folder_path = str(dir_nm) + '_fig/fig_full_picks/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass
    
    plt.figure()
    plt.imshow(normalize(data).T, cmap="seismic", vmin=-1, vmax=1, aspect="auto", extent=[x[0], x[-1], t[-1], t[0]], interpolation="none")
    color = picks["phase_type"].map({"P": "C0", "S": "C1"})
    plt.scatter(picks["channel_index"].values * dx, picks["phase_index"].values * dt, c=color, s=1)
    plt.scatter([], [], c="C0", label="P")
    plt.scatter([], [], c="C1", label="S")
    plt.legend()
    plt.xlabel("Distance (m)")
    plt.ylabel("Time (s)")
    plt.savefig(folder_path + BBB + '_picks.png')
    #plt.close()

def ploting_new(hdf5_filename, dir_nm):
    
    with h5py.File(hdf5_filename, "r") as fp:
        ds = fp["data"]
        data = ds[...]
        dt = ds.attrs["dt_s"]
        dx = ds.attrs["dx_m"]
        nx, nt = data.shape
        x = np.arange(nx) * dx
        t = np.arange(nt) * dt
        print(data.shape)
        #print(data)
        print(ds.attrs.keys())
        print([ds.attrs['dt_s'], ds.attrs['dx_m'], ds.attrs['unit']])
    
    AAA = hdf5_filename.split('.')[-2]
    picks = pd.read_csv(f"results/picks_phasenet_das/{AAA}.csv")
    BBB = hdf5_filename.split('.')[-2]
    BBB = BBB.split('/')[-1]
    
    folder_path = str(dir_nm) + '_fig/fig_full_raw/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass
    
    plt.figure()
    plt.imshow(normalize(data).T, cmap="seismic", vmin=-1, vmax=1, aspect="auto", extent=[x[0], x[-1], t[-1], t[0]], interpolation="none")
    plt.xlabel("Distance (m)")
    plt.ylabel("Time (s)")
    plt.savefig(folder_path + BBB + '.png')
    #plt.close()
    
    
    folder_path = str(dir_nm) + '_fig/fig_full_picks/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass
    
    plt.figure()
    plt.imshow(normalize(data).T, cmap="seismic", vmin=-1, vmax=1, aspect="auto", extent=[x[0], x[-1], t[-1], t[0]], interpolation="none")

    ############# Version 1 #############
    #picks_u = picks[picks["phase_type"].str.endswith("_u")]
    #picks_d = picks[picks["phase_type"].str.endswith("_d")]
    # 繪製數據（地上相位）
    #plt.scatter(picks_u["channel_index"].values * dx, 
                #picks_u["phase_index"].values * dt, 
                #c=picks_u["phase_type"].map({"P_u": "C0", "S_u": "C1"}), 
                #s=1, 
                #marker='_')  # 使用平行線標記
    # 繪製數據（地下相位）
    #plt.scatter(picks_d["channel_index"].values * dx, 
                #picks_d["phase_index"].values * dt, 
                #c=picks_d["phase_type"].map({"P_d": "C2", "S_d": "C3"}), 
                #s=1, 
                #marker='|')  # 使用垂直線標記
    
    ############# Version 2 #############
    color = picks["phase_type"].map({"P_u": "C0", "S_u": "C1", "P_d": "C2", "S_d": "C3"})
    plt.scatter(picks["channel_index"].values * dx, picks["phase_index"].values * dt, c=color, s=5)
    
    plt.scatter([], [], c="C0", label="P_u")
    plt.scatter([], [], c="C1", label="S_u")
    plt.scatter([], [], c="C2", label="P_d")
    plt.scatter([], [], c="C3", label="S_d")
    plt.legend()
    plt.xlabel("Distance (m)")
    plt.ylabel("Time (s)")
    plt.savefig(folder_path + BBB + '_picks.png')
    #plt.close()
    
    

def ploting_new02(hdf5_filename, dir_nm):
    
    with h5py.File(hdf5_filename, "r") as fp:
        ds = fp["data"]
        data = ds[...]
        dt = ds.attrs["dt_s"]
        dx = ds.attrs["dx_m"]
        nx, nt = data.shape
        x = np.arange(nx) * dx
        t = np.arange(nt) * dt
        print(data.shape)
        #print(data)
        print(ds.attrs.keys())
        print([ds.attrs['dt_s'], ds.attrs['dx_m'], ds.attrs['unit']])
    
    AAA = hdf5_filename.split('.')[-2]
    picks = pd.read_csv(f"results/picks_phasenet_das/{AAA}.csv")
    BBB = hdf5_filename.split('.')[-2]
    BBB = BBB.split('/')[-1]
    
    folder_path = str(dir_nm) + '_fig/fig_full_raw/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass
    
    plt.figure()
    plt.imshow(normalize(data).T, cmap="yen_cmap01", vmin=-1, vmax=1, aspect="auto", extent=[x[0], x[-1], t[-1], t[0]], interpolation="none")
    plt.xlabel("Distance (m)")
    plt.ylabel("Time (s)")
    plt.savefig(folder_path + BBB + '.png')
    #plt.close()
    
    
    folder_path = str(dir_nm) + '_fig/fig_full_picks/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass
    
    plt.figure()
    plt.imshow(normalize(data).T, cmap="yen_cmap01", vmin=-1, vmax=1, aspect="auto", extent=[x[0], x[-1], t[-1], t[0]], interpolation="none")

    ############# Version 1 #############
    #picks_u = picks[picks["phase_type"].str.endswith("_u")]
    #picks_d = picks[picks["phase_type"].str.endswith("_d")]
    # 繪製數據（地上相位）
    #plt.scatter(picks_u["channel_index"].values * dx, 
                #picks_u["phase_index"].values * dt, 
                #c=picks_u["phase_type"].map({"P_u": "C0", "S_u": "C1"}), 
                #s=1, 
                #marker='_')  # 使用平行線標記
    # 繪製數據（地下相位）
    #plt.scatter(picks_d["channel_index"].values * dx, 
                #picks_d["phase_index"].values * dt, 
                #c=picks_d["phase_type"].map({"P_d": "C2", "S_d": "C3"}), 
                #s=1, 
                #marker='|')  # 使用垂直線標記
    
    ############# Version 2 #############
    
    mycolors = ['#F4A7B9', '#BCA7F4', '#A7F4E2', '#E0F4A7']
    colormap_name='yen_cmap06'
    plt.colormaps.unregister(colormap_name)
    colormap = LinearSegmentedColormap.from_list('', mycolors)
    plt.colormaps.register(colormap, name=colormap_name)
    my_color_palette_new06=sns.color_palette(colormap_name, n_colors=24)

    
    mycolors = ['#F17C67', '#C167F1', '#67DCF1', '#97F167']
    colormap_name='yen_cmap07'
    plt.colormaps.unregister(colormap_name)
    colormap = LinearSegmentedColormap.from_list('', mycolors)
    plt.colormaps.register(colormap, name=colormap_name)
    my_color_palette_new07=sns.color_palette(colormap_name, n_colors=24)

    
    ####################################
    
    color_0 = my_color_palette_new06[0]
    color_1 = my_color_palette_new07[0]
    color_2 = my_color_palette_new06[23]
    color_3 = my_color_palette_new07[23]
    
    patch_0 = mpatches.Patch(color=color_0, label='P_u')
    patch_1 = mpatches.Patch(color=color_1, label='S_u')
    patch_2 = mpatches.Patch(color=color_2, label='P_d')
    patch_3 = mpatches.Patch(color=color_3, label='S_d')
    
    labels = ['P-wave u', 'S-wave u', 'P-wave d', 'S-wave d']
    #color_0 = (0.4453038424135514, 0.7008951308320329, 0.24461361014994232)
    #color_1 = (0.25524119682475044, 0.1504534455073841, 0.8439016667797455)
    #color_2 = (0.9287740009498608, 0.22523916140850805, 0.1363728882556483)
    #color_3 = (0.991285704593256, 0.8666639527783432, 0.13584639392089012)
   
    color = picks["phase_type"].map({"P_u": color_0, "S_u": color_1, "P_d": color_2, "S_d": color_3})
    plt.scatter(picks["channel_index"].values * dx, picks["phase_index"].values * dt, c=color, s=5)
    
    plt.scatter([], [], color=color_0, label="P_u")
    plt.scatter([], [], color=color_1, label="S_u")
    plt.scatter([], [], color=color_2, label="P_d")
    plt.scatter([], [], color=color_3, label="S_d")
    
    plt.legend(handles=[patch_0, patch_1, patch_2, patch_3],
               labels=labels,
               bbox_to_anchor = (0.08, 1.01, 1, 0.5),
               loc = 'lower left',
               ncols=4,
               #title='Arrival time', 
               fontsize = 8)
    
    plt.xlabel("Distance (m)")
    plt.ylabel("Time (s)")
    plt.savefig(folder_path + BBB + '_picks.png')
    #plt.close()