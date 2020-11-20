import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    def plot_joints(self, posedata, savepath=None, axis='y', showX=False, xlim=None):

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
                # df_i.plot(x='t',y='x', alpha=.5, ax=ax2, c='blue')

            axes[i].set_xlabel('Time [s]')
            axes[i].set_ylabel('Position [px]')
            axes[i].grid()
            axes[i].set_title(j)
            if xlim is not None:
                axes[i].set_xlim(xlim)

            #filters
            y = df_i[axis]; t = df_i['t']
            y_lowess = lowess(y, t, frac=30/len(y), it=2)
            y_savgol = signal.savgol_filter(y, 15, 3)
            # axes[i].plot(y_lowess[:,0], y_lowess[:,1], label='lowess', c='orange')
            axes[i].plot(t, y_savgol, c='r', label='Savgol', alpha=.5)


        plt.suptitle(posedata.filename.strip('.h5'), fontsize=15, y=1.0)
        plt.tight_layout()


        if savepath is not None:
            plt.savefig(os.path.join(savepath+posedata.filename.strip('.h5'))+'.jpg', dpi=300)
            plt.close('all')



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
            return Snew.interpolate(method='cubic')
        else:
            return Snew

    def minmax(self, df):
        medx = (df.min() + df.max())/2
        dfnorm = 2*(df-medx)/(df.max() - df.min())

        return dfnorm


class FilterData:
    def __init__(self, p_cutoff=0.6):
        self.p_cutoff = p_cutoff

    def get_filterdata(self, poses, joints=None):
        df = poses.get_joint_data(joints)
        dfout = pd.DataFrame(columns=joints)

        for j in joints:
            s = (df.loc[:,(j,['x','likelihood'])]).copy()
            s.columns = s.columns.droplevel(0)
            s.loc[s.likelihood < self.p_cutoff,'x'] = np.nan
            s.interpolate(method='spline', order=3, inplace=True)
            x = s.x #use x-trajectory
            x.dropna(inplace=True)

            #high pass filter
            sos_filt = butter(8, 0.25, 'highpass', fs=30, output='sos')
            x_filt = sosfiltfilt(sos_filt,x.values)
            x_filt = pd.Series(data=x_filt, index=x.index)
            x = x_filt.copy()

            #zscore and remove outliers
            NZ = Normalizer()
            x_filt_z = NZ.zscore(x)
            x_filt_z = NZ.removeOutliers(x_filt_z,interp=True)
            x = x_filt_z.copy()

            #interpolate with savgol filter to remove unwanted highfreq jumps
            # try:
            #     x_savgol = signal.savgol_filter(x.values, 15, 3)
            #     x_savgol = pd.Series(data=x_savgol, index=x.index)
            #     x = x_savgol.copy()
            # except:
            #     print('savgol filter fit failed')


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
