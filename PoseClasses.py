import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import re, json
from pathlib import Path
from scipy import stats, signal
from scipy.signal import decimate, butter, sosfiltfilt, find_peaks, savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
import itertools
import os
import matplotlib


class PoseData:
    def __init__(self, path, filename, fps=30):
        self.path = Path(path)
        self.filename = filename
        self.fps = fps

        self.poses = pd.read_hdf(self.path / self.filename) #read and store dataframe as attribute
        self.poses.index /= self.fps

    def get_joint_names(self):
        return list(self.poses.columns.get_level_values(1).unique())

    def get_duration(self):
        return np.round(len(self.poses)/self.fps)

    def get_joint_data(self, joints=None):
        if joints is None: #return all joints data
            df = self.poses.copy()
            df.columns = df.columns.droplevel(0)
            return df
        else:
            for j in joints:
                assert j in joints
            df = self.poses.loc[:, (slice(None), joints, slice(None))]
            df.columns = df.columns.droplevel(0)
            return df


#TO DO - create a PlotPoses Class where the attributes are the marker size, type of plot, etc.
class AnalyzePoses:

    def __init__(self, joints):
        self.joints = joints

    #plot raw data for each joint
    def plot_joints(self, posedata, savepath=None, axis='x', showX=False, xlim=None):

        #plot time-series for each joint
        Nj = len(self.joints)
        fig, axes = plt.subplots(nrows=Nj//2, ncols=2, figsize=(18,Nj*3))
        axes = axes.ravel()
        for i,j in enumerate(self.joints):
            df_i = posedata.get_joint_data(j).copy()
            df_i = df_i.loc[:,(j, ['x','y','likelihood'])]
            df_i.columns = df_i.columns.droplevel()
            df_i['t'] = df_i.index
            if axis=='y':
                axes[i].invert_yaxis()
            axes[i].scatter(x='t',y=axis, c='likelihood', cmap='cool', data=df_i, marker='^', alpha=.5, s=10)
            # df_i.plot(x='t',y=axis, alpha=.5, ax=axes[i])
            if showX is True and axis=='y':
                ax2 = axes[i].twinx()
                ax2.scatter(x='t',y='x', c='likelihood', cmap='cool', data=df_i, marker='x', alpha=.5)

            axes[i].set_xlabel('Time [s]')
            axes[i].set_ylabel('Position [px]')
            axes[i].grid()
            axes[i].set_title(j)
            if xlim is not None:
                axes[i].set_xlim(xlim)

        plt.legend()
        plt.suptitle(posedata.filename.strip('.h5'), fontsize=15, y=1.0)
        plt.tight_layout()

        if savepath is not None:
            plt.savefig(os.path.join(savepath+posedata.filename.strip('.h5'))+'.jpg', dpi=300)
            plt.close('all')


    #plot raw and filtered side by side
    def plot_raw_filtered(self, posedata, plot_spd=False):

        Nj = len(self.joints)
        fig, axes = plt.subplots(nrows=Nj, ncols=2, figsize=(18,Nj*4), sharex=True, sharey=False)
        axes = axes.ravel()

        #filtered data
        Filter = FilterData()
        poses_filt = Filter.get_filterdata(posedata,self.joints) #the filtered poses

        for i,j in enumerate(self.joints):
            df_i = posedata.get_joint_data(j).copy()
            df_i = df_i.loc[:,(j, ['x','y','likelihood'])]
            df_i.columns = df_i.columns.droplevel()
            df_i['t'] = df_i.index

            axes[i].scatter(x='t',y='x', c='likelihood', cmap='cool', data=df_i, marker='^', alpha=.5, s=10)
            axes[i].plot(df_i['t'], df_i['x'], alpha=.5, lineWidth=0.5)
            axes[i+2].scatter(poses_filt.index, poses_filt[j], alpha=.5, s=5)
            axes[i+2].plot(poses_filt.index, poses_filt[j], alpha=.5, lineWidth=0.5)

            axes[i].set_title(j); axes[i+2].set_title(j+' filtered')

            if plot_spd is True:
                spd = poses_filt[j].diff()
                ax2 = axes[i*2+1].twinx()
                ax2.plot(spd,c='k')

        for i,a in enumerate(axes):
            axes[i].grid()

        sns.despine()
        plt.tight_layout()



class FilterData:
    def __init__(self, p_cutoff=0.3):
        self.p_cutoff = p_cutoff

    def get_filterdata(self, poses, joints=None):
        df = poses.get_joint_data(joints)
        dfout = pd.DataFrame(columns=joints)

        for j in joints:
            s = (df.loc[:,(j,['x','likelihood'])]).copy()
            s.columns = s.columns.droplevel(0)
            s.loc[s.likelihood < self.p_cutoff,'x'] = np.nan
            # s.interpolate(method='spline', order=3, inplace=True)
            s.interpolate(inplace=True)
            x = s.x #use x-trajectory
            x.dropna(inplace=True)

            #high pass filter
            sos_filt = butter(8, 0.25, 'highpass', fs=30, output='sos')
            x_filt = sosfiltfilt(sos_filt,x.values)
            x_filt = pd.Series(data=x_filt, index=x.index)
            x = x_filt.copy()

            #remove remaining detection noise with median filter
            x_filt = x.rolling(5, center=True).median().dropna().interpolate()
            x = x_filt.copy()

            #zscore and remove outliers
            NZ = Normalizer()
            x_filt_z = NZ.zscore(x)
            x_filt_z = NZ.removeOutliers(x_filt_z,interp=True)
            x = x_filt_z.copy()

            #interpolate with savgol filter to remove unwanted highfreq jumps
            try:
                x_savgol = signal.savgol_filter(x.values, 15, 3)
                x_savgol = pd.Series(data=x_savgol, index=x.index)
                x = x_savgol.copy()
            except:
                print('savgol filter fit failed')


            # x_lowess = lowess(x.values, s.index.values, frac=30/len(x), it=2)
            # x_lowess = pd.Series(data=x_lowess[:,1], index=x_lowess[:,0])
            # x = x_lowess.copy()


            #remove detection noise with median filter
            # szf = x.rolling(4, center=True).median().interpolate().dropna()
            # x = szf.copy()
            # szf = sz.rolling(10, center=True).mean().interpolate().dropna()
            # szf = sz.rolling(10, center=True).median().rolling(10, center=True).mean().interpolate().dropna()
            # szf.plot(alpha=.5)

            dfout[j]=x

        return dfout



def compute_features(df):
    return df
    #extract x-y joints data


#class to iteratively load windows of data
#should avoid passing joints as an argument - rather should pass a pose object with a subset of joints
class DataLoader:
    def __init__(self, posedata, winsize, overlap, joints=None):
        self.posedata, self.winsize, self.overlap, self.joints = posedata, winsize, overlap, joints

    def __iter__(self):
        T = self.posedata.get_duration()
        step = self.winsize - (self.overlap*self.winsize)

        for i in np.arange(0, T, step): yield self.posedata.get_joint_data(self.joints)[i:i+step]

class FeatureExtractor:
    def __init__(self, dataloader, compute_fn=compute_features):
        self.dataloader, self.compute_fn = dataloader, compute_fn

    def __iter__(self):
        pass


class Normalizer:
    def __init__(self, mu=0, sigma=1):
        self.mu, self.sigma = mu, sigma

    def zscore(self, df, exclude=None):
#         df = df.select_dtypes(include='number').copy()
        df = (df - df.mean())/df.std()
        self.mu = df.mean; self.sigma = df.std()
        return df

    def removeOutliers(self, S, n_std=2, interp=True):

        mu = S.mean(); sigma = S.std()
        self.mu = mu; self.sigma = sigma

        Sz = S.copy()
        if sigma == 0:
            print('std dev = 0 - returning unmodified series')
            return Sz
        Sz = (Sz-mu)/sigma

        #remove outliers
        Snew = S.copy()
        Snew[np.abs(Sz) > n_std] = np.nan
        if interp is True:
            return Snew.interpolate(method='linear')
        else:
            return Snew

    def minmax(self, df):
        medx = (df.min() + df.max())/2
        dfnorm = 2*(df-medx)/(df.max() - df.min())

        return dfnorm



#Plotting functions
#input dataframe and labels and generate scatter plot of truth vs estimate SwSt
def scatterplot_2(data,x,y,ax,hue=None,legend=None, hue_order=None):
    tol = 0.1
    p = sns.scatterplot(x, y, data=data, hue=hue, legend=legend, alpha=.7, ax=ax, hue_order=hue_order)
    sns.despine()
    minval = min(min(data[x]),min(data[y])); maxval = max(max(data[x]), max(data[y]))
    ax.plot([minval,maxval],[minval, maxval],c='gray',linestyle='--', alpha=.5)
    return p

#NEW IMPLEMENTATION TBC
class Filter:
    def __init__(self, data):
        self.data = data

    def remove_low_p(self, x, p_cutoff=0.6):
        x_filt = x[x.likelihood < p_cutoff] = np.nan
        return x_filt

    def interpolate(self, x):
        x_filt = x.interpolate(method='spline', order=3)
        x_filt.dropna(inplace=True)
        return x_filt





def swing_stance(pksT,pksNT):

    minswingT = 0.05
    maxstanceT = 3

    T = list(pksT.values)+list(pksNT.values)
    S = list(np.ones_like(pksT))+list(np.zeros_like(pksNT))
    T_sorted = np.sort(T)
    inds_sorted = np.argsort(T)
    S = [S[i] for i in inds_sorted]
    #swings are 1; stances are -1; missed are 0
    swst = pd.DataFrame({'dT':np.diff(T_sorted), 'type':np.diff(S)})
    swst['Type'] = 'Missed'
    swst.loc[swst.type==1,'Type']='Swing'
    swst.loc[swst.type==-1,'Type']='Stance'
    swst.loc[(swst.dT > maxstanceT) | (swst.dT <minswingT),'Type'] = 'Missed'

    return swst


def findHSTO(szf, direction, plotdata=True):
    prom = .2
    if direction == 'L':
        szf*=-1 #invert peaks sign if walking towards the left
    pks,_ = find_peaks(szf, distance=10, prominence=prom)
    pksT = szf.index[pks]
    pksN,_ = find_peaks(szf*-1, distance=10, prominence=prom) #since signal is z-scored
    pksNT = szf.index[pksN]
    if plotdata is True:
        plt.figure()
        szf.plot()
        plt.plot(pksT,szf.iloc[pks],'x')
        plt.plot(pksNT,szf.iloc[pksN],'x',c='r')

    return pksT,pksNT

def updatename(x):
    if x == 'B':
        return 'Brace'
    elif x=='NB':
        return 'No Brace'
    else:
        return 'NA'


#parse AlphaPose json file and returns dataframe with poses
def json_to_csv_pose(filepath, video_size=(1280,720)):

#// Halpe 26 body keypoints
    d =     {0:  "Nose",
    1:  "LEye",
    2:  "REye",
    3:  "LEar",
    4:  "REar",
    5:  "LShoulder",
    6:  "RShoulder",
    7:  "LElbow",
    8:  "RElbow",
    9:  "LWrist",
    10: "RWrist",
    11: "LHip",
    12: "RHip",
    13: "LKnee",
    14: "RKnee",
    15: "LAnkle",
    16: "RAnkle",
    17:  "Head",
    18:  "Neck",
    19:  "Hip",
    20: "LBigToe",
    21: "RBigToe",
    22: "LSmallToe",
    23: "RSmallToe",
    24: "LHeel",
    25: "RHeel"}

    bodyparts = {v:k for k,v in d.items()}
    #the subset of bodyparts we want to extract and their array indices in OpenPose
    dict_bp = {}
    keys = ['RHip', 'RKnee', 'RAnkle', 'LHip','LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe',
          'RSmallToe','RHeel']
    values = [bodyparts.get(b) for b in keys]
    kv = zip(keys,values)
    for k,v in kv:
        dict_bp.update({k+'_x':v*3})
        dict_bp.update({k+'_y':v*3+1})
        dict_bp.update({k+'_c':v*3+2})

    #load pose files
    print(filepath)
    with open(filepath) as f:
        data=json.load(f)

    df = pd.DataFrame(data)
    df['box_dist_center'] = df.box.apply(lambda x: np.abs(x[0]+x[2]/2 - video_size[0]/2) )
    df['box_area'] = df.box.apply(lambda x:x[2]*x[3])
    df.reset_index(inplace=True) #to index rows

    #Heuristics to locate patient based on person id ('idx')
    #1. sort by id count (most present in video frames) and pick first two
    ids = (df.groupby('idx')['image_id'].count().sort_values(ascending=False).index[:2])
    #2. pick id closest to center between the 2
    patient_id = (df.groupby(['idx'])['box_dist_center'].mean()).loc[ids].sort_values().index[0]
    df = df.query('idx==@patient_id')
    df.reset_index(drop=True, inplace=True)


    #loop thru all frames keypoints
    poses = pd.DataFrame() #store poses for all frames
    for frame_nr, keypoints, box, box_area in zip(df.image_id, df.keypoints, df.box, df.box_area):
        frame_nr = int(frame_nr.split('.')[0]) #remove jpg extension
        d = {'Videoname':Path(filepath).stem, 'box':[box], 'box_area':box_area}
        #parse pose file for desired keypoints
        for k,v in dict_bp.items():
            d.update({k:keypoints[v]})
        #append
        poses = pd.concat((poses,pd.DataFrame(d, index=[frame_nr])))

    return poses

#convert video to frames
def FrameCapture(pathIn, pathOut):

    pathOut = Path(pathOut).as_posix()
    if not os.path.exists(pathOut):
        os.makedirs(pathOut, exist_ok=True)
    # Path to video file
    vidObj = cv2.VideoCapture(pathIn)
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read
        # function extract frames
        try:
            success, image = vidObj.read()
            # Saves the frames with frame-count

            cv2.imwrite(pathOut+"/%d.jpg" % count, image)
            count += 1
        except:
            print(success)

#convert pose dataframe from AlphaPose to DLC
def convert_to_DLC(poses, filename):
    newnames = ['Left Hip',
     'Left Knee',
     'Left Ankle',
     'Left Heel',
     'Left Toe',
     'Right Hip',
     'Right Knee',
     'Right Ankle',
     'Right Heel',
     'Right Toe']

    oldnames = ['LHip','LKnee','LAnkle','LHeel','LBigToe','RHip','RKnee','RAnkle','RHeel','RBigToe']

    df = pd.DataFrame()

    for old,new in zip(oldnames, newnames):
        cols0 = np.tile(filename,3)
        cols1 = np.tile([new],3)
        # print(old,new)
        cols2 = ['x','y','likelihood'] #new levels
        arrays = [np.array(cols0),np.array(cols1),np.array(cols2)]
        index = pd.MultiIndex.from_arrays(arrays, names=['filename','bodyparts','coords'])
        cols = [c for c in poses.columns if old in c]
        df_ = pd.DataFrame(poses[cols].values,  columns=index)
        df = pd.concat((df,df_),axis=1)

    return df
