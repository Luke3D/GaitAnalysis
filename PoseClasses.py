import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import re, json
from pathlib import Path
from scipy import stats, signal
from scipy.signal import decimate, butter, sosfiltfilt, find_peaks, savgol_filter
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import acf
from tqdm import tqdm
import itertools
import os
import matplotlib


class PoseData:
    def __init__(self, path, filename, fps=30):
        self.path = Path(path)
        self.filename = filename
        self.fps = fps

        poses = pd.read_hdf(self.path / self.filename) #read and store dataframe as attribute
        poses.columns = poses.columns.droplevel(0) #drop scorer name (added by DLC)
        self.poses = poses
        self.poses.index /= self.fps
        self.normalized = False #flag that indicates if pose coordinates have been scaled to leg length
        self.filtered = False #flag that indicates if coordinates have been filtered (smoothed)

    def get_joint_names(self):
        return list(self.poses.columns.get_level_values(0).unique())

    def get_duration(self):
        return np.round(len(self.poses)/self.fps)

    def get_joint_data(self, joints=None):
        if joints is None: #return all joints data
            df = self.poses.copy()
            return df
        else:
            for j in joints:
                assert j in joints
            df = self.poses.loc[:, (joints, slice(None))]
            return df


    def normalize_data(self):

        if self.normalized is True:
            print('data already scaled')
        else:

            df = self.get_joint_data()
            dfnorm = df.copy()

            #normalize pixel coords of each side by leg length and relative to center of the hips
            hip_center_x = (df.loc[:, ('Left Hip','x')] + df.loc[:, ('Right Hip','x')])/2
            hip_center_y = (df.loc[:, ('Left Hip','y')] + df.loc[:, ('Right Hip','y')])/2

            xref = hip_center_x; yref = hip_center_y #ref point

            #use leg length as a reference - use mean between left and right leg (can also use femur or shank)
            legL = df.loc[:,(['Left Hip','Left Ankle'],['x','y'])]
            legR = df.loc[:,(['Right Hip','Right Ankle'],['x','y'])]
            length_L = vector_norm(legL)
            length_R = vector_norm(legR)
            L = (length_L+length_R)/2

            for k in df.loc[:,(slice(None),['x'])].columns:
                dfnorm.loc[:,k] = (df.loc[:,k] - xref)/L
            for k in df.loc[:,(slice(None),['y'])].columns:
                dfnorm.loc[:,k] = (df.loc[:,k] - yref)/L

            self.poses = dfnorm
            self.normalized = True
            

    def filter_data(self, joints=None):
        pass
        
        # if self.normalized is True:
        #     print('data already scaled')
        # else:

        #     df = self.get_joint_data()
        #     if self.joints is None:
        #         joints = self.poses.get_joint_names()
        #     df_filt = pd.DataFrame(columns)


    def window_data(self, winsize=4, overlap=0.5):
        pass


    def __repr__(self):
        return '\n'.join([
            f'Path: {self.path}',
            f'Filename: {self.filename}',
            f'Video FPS: {self.fps}',
            f'Normalized: {self.normalized}',
            f'Filtered: {self.filtered}',
            ])

#create input (ts of keypoints) to train CNN
def CNN_create_dataset(path, filename, direction, gaitrite):

    joints = ['Left Toe', 'Right Toe', 'Left Heel', 'Right Heel', 'Left Ankle', 'Right Ankle', 'Left Knee', 'Right Knee']

    poses = PoseData(path, filename)

    #metadata
    subjid = '_'
    l = filename.split('_')[0:4]
    subjid = subjid.join(l)

    #filter data
    Filter = FilterData()
    ts = Filter.get_filterdata(poses,joints) #the filtered poses withouth confidence
    # ts = Filter.get_normalized_data(poses, joints) #raw values with confidence
    ts.index-=ts.index[0] #reset time to start from 0 (boundary)
    if direction =='L':
        ts*=-1 #mirror time series if walking towards left

    #additional features
    #distance between L and R ankle
    ts['LR_Ankle_dist'] = ts['Left Ankle'] - ts['Right Ankle']
    ts['LR_Knee_dist'] = ts['Left Knee'] - ts['Right Knee']

    poses_filtered = poses
    poses_filtered.poses = ts
    poses_filtered.filtered = True

    #window data
    DL = DataLoader(poses_filtered, winsize=4, overlap=0.75, joints=None)
    W = []
    for w in DL:
        if len(w) > DL.winsize*poses_filtered.fps: #to create clips of same size
            w = w.values
#             w = w.values[:, np.newaxis, :] # add extra dimension
            w = w[:-1,:] #to make windows even size
            W.append(w) 
    #assemble data for current subject
    x = np.stack(W, axis=0)

    return x, subjid


#plot and save figure for joint keypoint data for all videos
#input path of posefile data (.h5)
def save_plot_keypoint_ts(path, savepathfig, joints=None):

    if joints is None:
        joints = ['Left Toe', 'Right Toe', 'Left Heel', 'Right Heel', 'Left Ankle', 'Right Ankle']

    posefiles=[c for c in os.listdir(path) if c.endswith('.h5')]
    Filter = FilterData()
    print(f'saving figures into {savepathfig}')
    for filename in tqdm(posefiles):
        try:
            poses = PoseData(path, filename)
            y = Filter.get_filterdata(poses, joints)
            plt.figure(); y.plot()
            subjid = '_'
            l = filename.split('_')[0:4]
            subjid = subjid.join(l)
            plt.savefig(os.path.join(savepathfig,subjid+'.jpg'), dpi=300)
            plt.clf(); plt.close('all')
        except(IndexError):
            print(f'{filename} - metadata not found')




#plot keypoint time seires for AlphaPose and DLC
def compare_pose_outputs(filename, pathAP, pathDLC, joints=['Left Heel','Right Heel']):

    posefiles=[c for c in os.listdir(pathAP) if c.endswith('.h5')]
    print(filename)

    #plot with AlphaPose
    jj = joints
    poses = PoseData(pathAP, filename)
    PA = AnalyzePoses(jj)
#     PA.plot_joints(poses)
    PA.plot_raw_filtered(poses)

    name_to_search = Path(filename).stem
    filenameDLC = [f for f in os.listdir(pathDLC) if (name_to_search in f and f.endswith('.h5'))][0]
    poses = PoseData(pathDLC, filenameDLC)
    PA = AnalyzePoses(jj)
#     PA.plot_joints(poses)
    PA.plot_raw_filtered(poses)



#calculate vector norm for each frame from dataframe, and interpolates missing values
def vector_norm(df):
    v_norm = np.sqrt((df.loc[:,(slice(None),'x')].iloc[:,0] - df.loc[:,(slice(None),'x')].iloc[:,1])**2 +
        (df.loc[:,(slice(None),'y')].iloc[:,0] - df.loc[:,(slice(None),'y')].iloc[:,1])**2)
    v_norm.dropna(inplace=True)
    v_norm.interpolate(inplce=True)
    return v_norm



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


    def plot_filtered(self, posedata, hsto_truth=None, hsto_est=None):
        
        assert len(self.joints) == 2

        #plot L and R for each joint
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18,12), sharex=True, sharey=True)
        axes = axes.ravel()
        #filter data
        Filter = FilterData()
        poses_filt = Filter.get_filterdata(posedata, self.joints)
        #plot L and R on each col 
        for i,j in enumerate(self.joints):
            axes[i].scatter(poses_filt.index, poses_filt[j], alpha=.5, s=5)
            axes[i].plot(poses_filt.index, poses_filt[j], alpha=.5, lineWidth=2)
            axes[i].set_title(j+' filtered')
            #HS and TO truth (times)
            if hsto_truth is not None:
                    if 'Left' in j:
                        cols = ['LHS','LTO']
                    else:
                        cols = ['RHS','RTO']
                    for hs,to in zip(hsto_truth[cols[0]], hsto_truth[cols[1]]):
                        axes[i].axvline(x=float(hs),c='r', label='HS')
                        axes[i].axvline(x=float(to),c='g', label='TO')
            if hsto_est is not None:
                #parse side 
                side = j.split(' ')[0] #assumes first word is side
                assert side == 'Left' or side == 'Right'
                hs_all = hsto_est.query('Side==@side & Event=="HS"').Time
                to_all = hsto_est.query('Side==@side & Event=="TO"').Time
                for hs,to in zip(hs_all, to_all):
                        axes[i].axvline(x=float(hs),c='r', label='HS', linestyle='--')
                        axes[i].axvline(x=float(to),c='g', label='TO', linestyle='--')

        sns.despine()
        plt.tight_layout()
        plt.suptitle(posedata.filename.strip('.h5'), fontsize=12, y=1.0)


    #plot raw and filtered side by side
    def plot_raw_filtered(self, posedata, plot_spd=False, truth=None, est=None):

        Nj = len(self.joints)
        fig, axes = plt.subplots(nrows=Nj, ncols=2, figsize=(18,Nj*4), sharex=False, sharey=False)
        axes = axes.ravel()

        #filtered data
        Filter = FilterData()
        poses_filt = Filter.get_filterdata(posedata,self.joints) #the filtered poses

        for i,j in enumerate(self.joints):
            df_i = posedata.get_joint_data(j).copy()
            df_i = df_i.loc[:,(j, ['x','y','likelihood'])]
            df_i.columns = df_i.columns.droplevel()
            df_i['t'] = df_i.index

            axes[i*2].scatter(x='t',y='x', c='likelihood', cmap='cool', data=df_i, marker='^', alpha=.5, s=8)
            axes[i*2].plot(df_i['t'], df_i['x'], alpha=.5, lineWidth=2)
            axes[i*2+1].scatter(poses_filt.index, poses_filt[j], alpha=.5, s=5)
            axes[i*2+1].plot(poses_filt.index, poses_filt[j], alpha=.5, lineWidth=2)
            labels = axes[i*2+1].get_xticks()

            axes[i*2].set_title(j); axes[i*2+1].set_title(j+' filtered')
            axes[i].set_xlabel('Time [s]'); axes[i*2+1].set_xlabel('Time [s]')
            axes[i*2].set_ylabel('x-position [px]'); axes[i*2+1].set_ylabel('Normalized x-position [AU]')
            
            if plot_spd is True:
                spd = poses_filt[j].diff()
                ax2 = axes[i*2+1].twinx()
                ax2.plot(spd,c='k')

            if truth is not None:
                if 'Left' in j:
                    cols = ['LHS','LTO']
                else:
                    cols = ['RHS','RTO']
                for hs,to in zip(truth[cols[0]], truth[cols[1]]):
                    axes[i*2].axvline(x=float(hs),c='r', label='HS', lineWidth=1)
                    axes[i*2].axvline(x=float(to),c='g', label='TO', lineWidth=1)
                    axes[i*2+1].axvline(x=float(hs),c='r', label='HS', lineWidth=1)
                    axes[i*2+1].axvline(x=float(to),c='g', label='TO', lineWidth=1)

        # for a in axes:
        #     a[i].grid()
        #     a.set_xticks(df_i.index)

        sns.despine()
        plt.tight_layout()
        plt.suptitle(posedata.filename.strip('.h5'), fontsize=15, y=1.0)




class FilterData:
    def __init__(self, p_cutoff=0.3):
        self.p_cutoff = p_cutoff

    def get_normalized_data(self, poses, joints=None):
        df = poses.get_joint_data(joints)
        df_cols = joints + ([j+'_conf' for j in joints])
        dfout = pd.DataFrame(columns=df_cols)

        for j in joints:
            s = (df.loc[:,(j,['x','likelihood'])]).copy()
            s.columns = s.columns.droplevel(0)
            s.interpolate(inplace=True, method='linear')
            s.dropna(inplace=True)
            x = s.x #x-coord keypoint
            conf = s.likelihood #keypoint confidence

            #high pass filter
            sos_filt = butter(8, 0.25, 'highpass', fs=30, output='sos')
            x_filt = sosfiltfilt(sos_filt, x.values)
            x_filt = pd.Series(data=x_filt, index=x.index)
            x = x_filt.copy()

            #zscore (this is just another rescaling between -1,1 really). Optionally try removing outliers
            NZ = Normalizer()
            x_filt_z = NZ.zscore(x)
            x = x_filt_z.copy()

            #assemble keypoint + confidence
            dfout[j]=x
            dfout[j+'_conf'] = conf
        
        return dfout
        

    def get_filterdata(self, poses, joints=None):
        df = poses.get_joint_data(joints)
        dfout = pd.DataFrame(columns=joints)

        for j in joints:
            s = (df.loc[:,(j,['x','likelihood'])]).copy()
            s.columns = s.columns.droplevel(0)
            s.loc[s.likelihood < self.p_cutoff,'x'] = np.nan
            # s.interpolate(method='spline', order=3, inplace=True)
            s.interpolate(inplace=True, method='linear')
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

            #zscore (this is just another rescaling between -1,1 really). Optionally try removing outliers
            NZ = Normalizer()
            x_filt_z = NZ.zscore(x)
            # x_filt_z = NZ.removeOutliers(x_filt_z,interp=True)
            x = x_filt_z.copy()

            #interpolate with savgol filter to remove unwanted highfreq jumps
            # try:
            #     x_savgol = signal.savgol_filter(x.values, 15, 3)
            #     x_savgol = pd.Series(data=x_savgol, index=x.index)
            #     x = x_savgol.copy()
            # except:
            #     print('savgol filter fit failed')

            #interpolate with Gaussian Filter
            x_filt = gaussian_filter1d(x, sigma=1)
            x = pd.Series(data=x_filt, index=x.index)


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


#compute stats features on a window of data
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
        for i in np.arange(0, T, step): 
            # yield (self.posedata.get_joint_data(self.joints)[i: i+self.winsize], (i,i+self.winsize))
            yield self.posedata.get_joint_data(self.joints)[i: i+self.winsize]


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
    # p = sns.scatterplot(x, y, data=data, hue=hue, legend=legend, alpha=.7, ax=ax, hue_order=hue_order)
    p = sns.regplot(x, y, data=data, ax=ax)
    sns.despine()
    minval = min(min(data[x]),min(data[y])); maxval = max(max(data[x]), max(data[y]))
    ax.plot([minval,maxval],[minval, maxval],c='gray',linestyle='--', alpha=.5)
    xydata = data[[x,y]].copy()
    xydata.dropna(inplace=True)
    r,rp = pearsonr(xydata[x],xydata[y])
    # print(sum(data[x].isnull()), sum(data[y].isnull()))
    ax.set_title(f'r={r:.3f}')
    plt.tight_layout()
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

    T = list(pksT.values)+list(pksNT.values) #peak times
    S = list(np.ones_like(pksT))+list(np.zeros_like(pksNT)) #binary indicator of peak and trough
    T_sorted = np.sort(T) #sort the time sequence
    inds_sorted = np.argsort(T) #sort the array index sequence
    S = [S[i] for i in inds_sorted] #the binary array indicating HS and TO
    #swings are 1; stances are -1; missed are 0
    swst = pd.DataFrame({'dT':np.diff(T_sorted), 'type':np.diff(S)})
    swst['Type'] = 'Missed'
    swst.loc[swst.type==1,'Type']='Swing'
    swst.loc[swst.type==-1,'Type']='Stance'
    swst.loc[(swst.dT > maxstanceT) | (swst.dT <minswingT),'Type'] = 'Missed'

    return swst



# #returns sequence of DST for each step
#inputs dataframe y of left and right leg keypoint timeseries
def gait_params(y, direction, plotdata=False):

    #define dictionary of parameters
    gait_par = {'HSTO':[], 'swingL':[], 'stanceL':[], 'swingR':[], 'stanceR':[],
                'DST':[], 'asymmetry_swing':[], 'asymmetry_stance':[],
                'cadence':[], 'step_F_Left':[], 'step_F_Right':[], 'swing_stance':[]}

    minswingT = 0.05
    maxstanceT = 3

    #HS and TO for each side
    HSTO = pd.DataFrame()
    for side_j in y.columns:
        y_side = y[side_j]
        side = side_j.split(' ')[0]
        HS,TO = findHSTO(y_side, direction, plotdata=False) #find positive and neg peaks (approx HS, TO)
        T = list(HS.values)+list(TO.values) #HS,TO times for current side
        S = ['HS' for i in range(len(HS))] + ['TO' for i in range(len(TO))]
        T_sorted = np.sort(T) #sort the time sequence
        inds_sorted = np.argsort(T) #sort the array index sequence
        S = [S[i] for i in inds_sorted] #array with HS TO names

        HSTO = pd.concat((HSTO, pd.DataFrame({'Event':S, 'Side':side}, index=T_sorted)))
    #add binary value to identify HS and TO and sort by event time
    HSTO['Event_binary'] = 0
    HSTO.loc[HSTO.Event=='HS','Event_binary'] = 1
    HSTO = HSTO.sort_index()
    gait_par['HSTO'] = HSTO

    #compute swing and stance
    swst = pd.DataFrame()
    for side in HSTO.Side.unique():
        HSTO_side = HSTO.query('Side==@side')
        swst = pd.concat((swst, pd.DataFrame({'dT':np.diff(HSTO_side.index),
        'Event_binary':np.diff(HSTO_side.Event_binary), 'side':side})), axis=0)
        swst['Type'] = 'Missed'
        swst.loc[swst.Event_binary==-1,'Type']='Stance'
        swst.loc[swst.Event_binary==1,'Type']='Swing'
        swst.loc[(swst.dT > maxstanceT) | (swst.dT <minswingT),'Type'] = 'Missed'
    gait_par['stanceR'] = swst.query('Type=="Stance" & side=="Right"').dT.median()
    gait_par['swingR'] = swst.query('Type=="Swing" & side=="Right"').dT.median()
    gait_par['stanceL'] = swst.query('Type=="Stance" & side=="Left"').dT.median()
    gait_par['swingL'] = swst.query('Type=="Swing" & side=="Left"').dT.median()
    # gait_par['swing_stance'] = swst

    #asymmetry index swing and stances
    asym_swing = asymmetry_index(swst, type='Swing')
    asym_stance = asymmetry_index(swst, type='Stance')
    gait_par['asymmetry_swing'] = asym_swing
    gait_par['asymmetry_stance'] = asym_stance

    #DST
    DST = pd.DataFrame()
    #L Double support
    Lhs = HSTO.loc[(HSTO.Side=='Left') & (HSTO.Event=='HS')].index
    Rto = HSTO.loc[(HSTO.Side=='Right') & (HSTO.Event=='TO')].index
    Lhs = np.sort(Lhs)
    Rto = np.sort(Rto)
    #match closest preceding HS with each TO
    Rto_Lhs = []
    for i in Rto:
        indmin = np.argmin(np.abs(Lhs - i))
        Rto_Lhs.append(np.array([i, Lhs[indmin]]))
    Rto_Lhs = np.stack(Rto_Lhs, axis=0)
    DST_L = pd.concat((DST, pd.DataFrame({'DST':Rto_Lhs[:,0] - Rto_Lhs[:,1], 'Side':'Left'})), axis=0)
   #R Double support
    Rhs = HSTO.loc[(HSTO.Side=='Right') & (HSTO.Event=='HS')].index
    Lto = HSTO.loc[(HSTO.Side=='Left') & (HSTO.Event=='TO')].index
    Rhs = np.sort(Rhs)
    Lto = np.sort(Lto)
    #match closest preceding HS with each TO
    Lto_Rhs = []
    for i in Lto:
        indmin = np.argmin(np.abs(Rhs - i))
        Lto_Rhs.append(np.array([i, Rhs[indmin]]))
    Lto_Rhs = np.stack(Lto_Rhs, axis=0)
    DST_R = pd.concat((DST, pd.DataFrame({'DST':Lto_Rhs[:,0] - Lto_Rhs[:,1], 'Side':'Right'})), axis=0)
    DST = (DST_L.DST + DST_R.DST).median()
    gait_par['DST'] = DST

    #cadence - we can use autocorrelation or count steps
    #autocorrelation
    for side in y.columns:
        y_side = y[side]
        # print(y_side.shape)
        ac, ci = acf(y_side, nlags=len(y_side)//2, alpha=.05)
        t = np.arange(0, len(ac)/30, 1/30)
        # plt.plot(t,ac)
        pks, _ = find_peaks(ac, distance=10)
        stepf = pks[0]/30*60 #steps/min
        gait_par['step_F_'+side.split(' ')[0]] = stepf


    #calculate number of HS / time
    N_steps = len(HSTO.query('Event=="HS"'))
    t = HSTO.query('Event=="HS"').index
    T = t[-1] - t[0]
    gait_par['cadence'] = N_steps/T*60

    return gait_par


# Asymmetry index - assumes dataframe with dT, side and type
# (L-R) / (L+R) * 100
def asymmetry_index(swst, type='Swing'):
    L = swst.query('side=="Left" & Type==@type')
    R = swst.query('side=="Right" & Type==@type')
    asym_idx = np.abs(L.dT.median() - R.dT.median()) / (L.dT.median() + R.dT.median())*100
    return asym_idx



#inputs a normalized time series of joint trajectory and returns positive and negative peak times
def findHSTO(szf, direction, plotdata=True, ax=None):
    prom = 1
    if direction == 'L':
        szf*=-1 #invert peaks sign if walking towards the left
    pks,_ = find_peaks(szf, distance=10, prominence=prom) #return peaks index
    pksT = szf.index[pks] #convert index to peak time
    pksN,_ = find_peaks(szf*-1, distance=10, prominence=prom) #find negative peaks
    pksNT = szf.index[pksN]
    if plotdata is True:
        if ax is None:
            fig, ax = plt.subplots(1,1)
        szf.plot(ax=ax)
        ax.plot(pksT,szf.iloc[pks],'x', c='r') #HS
        ax.plot(pksNT,szf.iloc[pksN],'o',c='r') #TO

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
