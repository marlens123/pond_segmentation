import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pyproj
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from numba import njit
from glob import glob
import time

# Geomapping based on https://github.com/peterkovesi/ImageProjectiveGeometry.jl (projective.jl, function imagept2plane)

# --- Filenames:
mss_file = '/Users/gspreen/Data/Bathymetry-SSH/SeaSurfaceHeight/DTU21MSS_1min.mss.nc'

# flight 1: test
#cam_file1 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/220705_105319/IRdata_ATWAICE_220705_105319.nc'
#apx_file1 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220705_01_PS131_Heli-PS/insgps-atwaice-pospac_export_gnssprim_20220705_01_CP780_098.txt'
#save_nc_file1 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/220705_105319/IRdata_ATWAICE_processed_220705_105319.nc'
# flight 2: test with ALS
#cam_file2 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/220707_130106/IRdata_ATWAICE_220707_130106.nc'
#apx_file2 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220707_02_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220707_02.txt'
#save_nc_file2 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/220707_130106/processed_flight_data.nc'
# flight 3: floe 1 (IR and Canon)
#cam_file3 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/220713_075354/IRdata_ATWAICE_220713_075354.nc'
#apx_file3 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220713_01_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220713_01.txt'
#save_nc_file3 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/220713_075354/IRdata_ATWAICE_processed_220713_075354.nc'
# flight 4: floe 1 (IR)
#cam_file4 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight04_220713_104532/IRdata_ATWAICE_220713_104532.nc'
#apx_file4 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220713_02_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220713_02.txt'
#save_nc_file4 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight04_220713_104532/IRdata_ATWAICE_processed_220713_104532.nc'
# flight 6 EM-bird (only parts of INS data available -> use GPS data from IR camera in future)
#cam_file6 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight06_220717_075915/IRdata_ATWAICE_220717_075915.nc'
#apx_file6 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220717_01_PS131_Heli-PS/insgps-ps131-pospac_export_20220717_01.txt'
#save_nc_file6 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight06_220717_075915/IRdata_ATWAICE_processed_220717_075915.nc'
# flight 7
#cam_file7 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight07_220717_122355/IRdata_ATWAICE_220717_122355.nc'
#apx_file7 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220717_04_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220717_04.txt'
#save_nc_file7 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight07_220717_122355/IRdata_ATWAICE_processed_220717_122355.nc'
# flight 8
#cam_file8 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight08_220718_081257/IRdata_ATWAICE_220718_081257.nc'
#apx_file8 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220718_01_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220718_01.txt'
#save_nc_file8 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight08_220718_081257/IRdata_ATWAICE_processed_220718_081257.nc'
# flight 9
cam_file9 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight09_220718_142920/IRdata_ATWAICE_220718_142920.nc'
apx_file9 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220718_02_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220718_02.txt'
save_nc_file9 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight09_220718_142920/IRdata_ATWAICE_processed_220718_142920.nc'
# flight 10 (EM-bird)
#cam_file10 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight10_220719_070453/IRdata_ATWAICE_220719_070453.nc'
## Missing: apx_file = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/'
#save_nc_file10 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight10_220719_070453/IRdata_ATWAICE_processed_220719_070453.nc'
# flight 11-1: transect to floe 2, 1000ft, 80kn
#cam_file11 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight11_220719_104906/IRdata_ATWAICE_220719_104906.nc'
#apx_file11 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220719_02_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220719_02.txt'
#save_nc_file11 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight11_220719_104906/IRdata_ATWAICE_processed_220719_104906.nc'
# flight 11-2: floe 2 grid 500ft
#cam_file11 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight11_220719_112046/IRdata_ATWAICE_220719_112046.nc'
#apx_file11 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220719_02_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220719_02.txt'
#save_nc_file11 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight11_220719_112046/IRdata_ATWAICE_processed_220719_112046.nc'
# flight 11-3: floe 2 grid 1000ft + transect
#cam_file11 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight11_220719_114341/IRdata_ATWAICE_220719_114341.nc'
#apx_file11 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220719_02_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220719_02.txt'
#save_nc_file11 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight11_220719_114341/IRdata_ATWAICE_processed_220719_114341.nc'
# flight 13
#cam_file13 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight13_220724_131311/IRdata_ATWAICE_220724_131311.nc'
#apx_file13 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220724_01_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220724_01.txt'
#save_nc_file13 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight13_220724_131311/IRdata_ATWAICE_processed_220724_131311.nc'
# flight 14: floe 1 grid + transect
#cam_file14 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight14_220730_042852/IRdata_ATWAICE_220730_042841.nc'
#apx_file14 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220730_01_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220730_01.txt'
#save_nc_file14 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight14_220730_042852/IRdata_ATWAICE_processed_220730_042841.nc'
# flight 15: EM-bird
#cam_file15 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight15_220730_085252/IRdata_ATWAICE_220730_085252.nc'
#apx_file15 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220730_02_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220730_02.txt'
#save_nc_file15 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight15_220730_085252/IRdata_ATWAICE_processed_220730_085252.nc'
# flight 16
#cam_file16 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight16_220730_111439/IRdata_ATWAICE_220730_111439.nc'
#apx_file16 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220730_03_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220730_03.txt'
#save_nc_file16 = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight16_220730_111439/IRdata_ATWAICE_processed_220730_111439.nc'
# flight 17
#cam_file = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight17_220808_084908/IRdata_ATWAICE_220808_084908.nc'
#apx_file = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/gpsins-ascii/20220808_01_PS131_Heli-PS/insgps-ps131-pospac_export_gnssprim_20220808_01.txt'
#save_nc_file = '/Users/gspreen/Seafile/Gunnar/Campaigns/2022-07_PS131_ATWAICE/Data/ir-camera/Flight17_220808_084908/IRdata_ATWAICE_processed_220808_084908.nc'

nth=4 # only every nth file is taken to reduce dataset size 8and memory usage)

###########################
# -------FUNCTIONS------- #
###########################

def image_gradient(T_arr):
    """
    For operational use, included in the function.
    - input:
        T_arr: numpy.array
    - output:
        T_grad: numpy.array
    """
    T_diff=np.zeros((480,640))
    cc=0
    p10=np.percentile(T_arr,25)
    for T in T_arr:
        if np.mean(T)<=p10:
            T_diff += T/T[int(480/2),int(640/2)]
            cc+=1
    T_grad = T_diff/cc
    print('Number of images used for empirical gradient: ', cc) 
    
    return T_grad

def nuc_correction(var, i):
    """   
    detecting NUC and remove frame before and NUC frames for MOSAiC default of 4fps
    the idea is the following when nuc apears this freezen the screen
    image for some time. That does not mean that with 4fps one complete
    frame is frozen since the length of the nuc is not always the same.
    This decets if within the last and next image a couple for lines close
    to the previous one are the same or very similar. If this is the case,
    either the helicopter was very slow (unlikely since the groundspeed for
    laser scanner flights is 90knots) or there was a NUC if there
    was a NUC in any of the frames, these should be skipped because the
    location from the INS data will be off by some pixels which will be
    visible in the mosaic.
    """
    
    vm=var[i-1][225:255,:]
    v1=var[i][225:255,:]
    v2=var[i+1][225:255,:]
    nuctest_forward=np.abs(v1-v2).mean(axis=1).min()
    nuctest_backward=np.abs(v1-vm).mean(axis=1).min()
    # print(len(proclist),nuctest_forward,nuctest_backward)
    # print (i,nuctest)
    if nuctest_forward==0.0: 
        #a difference of 0 means that the frames must
        #be identical, this means thta the second frame i.e., i+1 is still
        #during the NUC, thus is should be skipped as well
        i+=1

    return i

def get_corrds(lon,lat,lon0,lat0):
    proj_dict_new = dict(proj='stere', 
                         lat_0=lat0,
                         lat_ts=lat0,
                         lon_0=lon0,
                         k=1,
                         x_0=0,
                         y_0=0,
                         ellps='WGS84',
                         units='m')
    p = pyproj.Proj(proj_dict_new)
    
    return p(lon,lat)


def get_mss(lat, lon, mss, lon_mss, lat_mss):
    """
    Loading Mean sea surface height data. Calculating values along the flight track.
    Input
        - lon, lat: numpy.ndarray
        - mss, lon_mss, lat_mss: pd.Series
    Output
        - mss_ts: list
    """
    
    # getting index of closest pixel of mss data to flight track
    mss_ts = []
    ix_lat = np.abs(lat_mss-lat).argmin()
    ix_lon = np.abs(lon_mss-lon).argmin()
    mss_ts.append(mss[ix_lat,ix_lon])
    
    return mss_ts[0]


@njit
def rotx(r):
    Rr = np.array([[1.0, 0.0, 0.0, 0.0],[0.0, np.cos(r), -np.sin(r), 0.0],[0.0, np.sin(r), np.cos(r), 0.0],[0.0, 0.0, 0.0, 1.0]])
    return Rr

@njit
def roty(p):
    Rp = np.array([[np.cos(p), 0.0, np.sin(p), 0.0],[0.0, 1.0, 0.0, 0.0],[-np.sin(p), 0.0, np.cos(p), 0.0],[0.0, 0.0, 0.0, 1.0]])
    return Rp

@njit
def rotz(y):
    Ry = np.array([[np.cos(y), -np.sin(y), 0.0, 0.0],[np.sin(y), np.cos(y), 0.0, 0.0],[0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0]])
    return Ry

@njit
def pix_loc(roll,pitch,yaw,height,xc,yc,p_off,ho,x_coord,y_coord):

    ro,po,yo,f,k1 = p_off

    # Rotation matrix
    pxrot=rotx(np.deg2rad(roll)) 
    pyrot=roty(np.deg2rad(pitch))
    pzrot=rotz(np.deg2rad(yaw))

    # Rotation
    crot=(pxrot@rotx(np.deg2rad(ro)))@(pyrot@roty(np.deg2rad(po)))@(pzrot@rotz(np.deg2rad(yo)))
    Rc=crot[0:3,0:3] # rotation  matrix
    
    # New camera location
    camera_location = np.array([xc,yc,height-ho]) 
   
    for yi in range(640):
        for xi in range(480):

            # normalization pixel positions
            x_d = (xi-240)/f
            y_d = (yi-320)/f

            # radial distortion factor
            rsqrd = x_d**2 + y_d**2
            r_d = 1 + k1*rsqrd 

            # undistorted pixel positions
            x_n = x_d/r_d
            y_n = y_d/r_d
            
            # Rotate viewing farme to world frame
            ray_tmp = np.array([x_n, y_n, 1.])   
            ray = Rc.T@ray_tmp     
#             ray = mul(Rc.T,ray_tmp)
            k = -camera_location[-1]/ray[-1]

            # new point of intersection coordinates
            pt = camera_location + k*ray
            
            # write point to array - coordinates of full image
            x_coord[xi,yi]=pt[0]
            y_coord[xi,yi]=pt[1]
            
    return x_coord,y_coord

###########################
# -------DATA READ------- #
###########################
print('Load data:')
############
# --- IR Cam
############
cam_data = xr.open_dataset(cam_file) # ADJUST FILE NAME HERE

# time
date_str = cam_file[-16:-3]
starttime = datetime.strptime(date_str, '%y%m%d_%H%M%S')
date = datetime.strptime(date_str[:-7], '%y%m%d')
init = (starttime-date).total_seconds()
ttt = init + .25 * np.arange(0,len(cam_data.time))

# images with new time
cam_data = cam_data.assign_coords(time=ttt)
temps = cam_data.temps #[:1000] # <------------------ONLY SUBSET OF DATA, DEFINE RANGE HERE (for reducing required memory)
temps = temps[::nth] # here only every fourth image will be taken/reduced frequency (for reducing required memory)

# radial image gradient
gradient_array = image_gradient(temps.data)

##############
# --- Applanix
##############
df_apx = pd.read_csv(apx_file,skiprows=27,sep='\s+', header=None)
header = pd.read_csv(apx_file, header=23,nrows=0,sep=',\s', engine='python')
df_apx.columns=list(header)
df_apx['TIME']=(df_apx['TIME']%86400)
df_apx = df_apx.set_index('TIME')

# Interpolators for Applanix variables (to be fitted with Camera times)
int_lon = interp1d(df_apx.index, df_apx.LONGITUDE)
int_lat = interp1d(df_apx.index, df_apx.LATITUDE)
int_roll = interp1d(df_apx.index, df_apx.ROLL)
int_pitch = interp1d(df_apx.index, df_apx.PITCH)
int_heading = interp1d(df_apx.index, df_apx.HEADING)
int_height = interp1d(df_apx.index, df_apx['ELLIPSOID HEIGHT'])

mlon = float(np.mean(df_apx.LONGITUDE))
mlat = float(np.mean(df_apx.LATITUDE))

# mean sea surface height
data_mss = Dataset(mss_file)
mss = data_mss.variables['mss'][:]
lat_mss=data_mss.variables['lat'][:]
lon_mss=data_mss.variables['lon'][:]

################
# --- PROCESSING
################
print('Processing:')

img_ix_list=[];img_list=[];xd_list=[];yd_list=[]
time_list=[];lon_list=[];lat_list=[]
roll_list=[];pitch_list=[];mod_head_list=[];
alti_list=[];mss_list=[]

c=0
emissivity=0.996
x_coord=np.zeros((480,640));y_coord=np.zeros((480,640))

start=time.time()
print('Image georef ...')
for ix in range(len(temps)-1):
    if ix==int(len(temps)/2):
        print('Half way!')

    # NUC correction
    i = nuc_correction(temps, ix)

    # not all images have a allocated Apx value
    try:
        # read values for image time
        ttime=float(temps[i].time)

        img=(temps[i].values)/100
        #img = img[::-1,::-1] # <--------- Here image orientation is adapted
        #img = img[:,:] # <--------- Here image orientation is adapted
        img = img[::,::]

        lon=float(int_lon(ttime))
        lat=float(int_lat(ttime))
        roll=float(int_roll(ttime))
        pitch=float(int_pitch(ttime))
        heading=float(int_heading(ttime))
        altitude=float(int_height(ttime))

        # heading from [0,360] to [-180,180]
        mod_head = heading + (mlon-lon)
        if mod_head<-180: 
            mod_head+=360
        elif mod_head>180: 
            mod_head-=360
        else:
            pass 

        # correction of image gradient
        img = img/gradient_array.data
        img = img/emissivity

        # Mean sea surface height at location for height correction
        p_mss = get_mss(lat, lon, mss, lon_mss, lat_mss)

        # Stereographic x,y coordinates in meter
        xloc,yloc=get_corrds(lon,lat,mlon,mlat)
    
        # image pixels
        voff=(0.458456,0.134936,88.3343,604.362,-0.407934) # Fit parameter for MOSAiC Leg 1 mit (roll,pitch,yaw,f,k)
        xd,yd = pix_loc(roll,-pitch,mod_head,altitude,xloc,yloc,voff,p_mss,x_coord,y_coord)

        # object assignment, because otherwise redirected and the same
        xd_tmp = xd.copy() 
        yd_tmp = yd.copy()

        # write values/arrays to list to save them to a nc file
        img_ix_list.append(i)
        img_list.append(img)
        time_list.append(ttime)
        roll_list.append(roll)
        pitch_list.append(pitch)
        alti_list.append(altitude)
        lon_list.append(lon)
        lat_list.append(lat)
        xd_list.append(xd_tmp)
        yd_list.append(yd_tmp)
        mod_head_list.append(mod_head)
        mss_list.append(p_mss)

    except ValueError:
        c+=1

print(str(c)+' values are not in interpolation range')
end = time.time()
print(str(round((end-start)/60))+' minutes')
print('Save NetCDF:')
print(str(len(img_list))+' processed images included.')

#############################
# -------SAVE NETCDF------- #
#############################

rootgrp = Dataset(save_nc_file, "w", format="NETCDF4") # adjust filename to specific flight details

# --- General
rootgrp.Title="UB IR Camera images"
rootgrp.Institution="University of Bremen"
rootgrp.PI_name="Dr. Gunnar Spreen"
rootgrp.Product_name="Thermal infrared sea ice surface temperature images"
rootgrp.Description="This data is processed during the PS131 ATWAICE Campaign."
rootgrp.Date=str(date)

# --- Dimensions:
ncx = rootgrp.createDimension("x", 480)
ncy = rootgrp.createDimension("y", 640)
ncstep = rootgrp.createDimension("t", len(img_list))

# --- Variables:

crs=rootgrp.createVariable("crs","c") 
crs.grid_mapping_name = "stereographic"
crs.latitude_of_projection_origin = mlat
crs.longitude_of_projection_origin = mlon
crs.reference_ellipsoid_name='WGS84'

nctemp=rootgrp.createVariable("Ts","f4",("t","x","y"),zlib=True)
nctemp.units = "K"
nctemp.standard_name = "Surface temperature"
nctemp.grid_mapping = "crs"

nclon = rootgrp.createVariable("lon","f4",("t"),zlib=True)
nclon.units = "degrees_east"
nclon.standard_name = "longitude"

nclat = rootgrp.createVariable("lat","f4",("t"),zlib=True)
nclat.units = "degrees_north"
nclat.standard_name = "latitude"    

nctimes=rootgrp.createVariable("time","f4",("t"),zlib=True)
nctimes.units = "seconds since "+str(date)[:-9]+" 00:00:00"
nctimes.standard_name = "time"

ncroll = rootgrp.createVariable("roll","f4",("t"),zlib=True)
ncroll.units = "degree"
ncroll.standard_name = "Roll angle"

ncpitch = rootgrp.createVariable("pitch","f4",("t"),zlib=True)
ncpitch.units = "degree"
ncpitch.standard_name = "Pitch angle"

ncalti = rootgrp.createVariable("altitude","f4",("t"),zlib=True)
ncalti.units = "m"
ncalti.standard_name = "Altitude"

nciix = rootgrp.createVariable("ix","f4",("t"),zlib=True)
nciix.units = "1"
nciix.standard_name = "Image index"

nchead = rootgrp.createVariable("heading","f4",("t"),zlib=True)
nchead.units = "degree"
nchead.standard_name = "Modified heading"

ncmss = rootgrp.createVariable("mss","f4",("t"),zlib=True)
ncmss.units = "m"
ncmss.standard_name = "Mean sea surface height"

ncxd=rootgrp.createVariable("xd","f4",("t","x","y"),zlib=True)
ncxd.units = "m"
ncxd.standard_name = "x coordinate"
ncxd.grid_mapping = "crs"

ncyd=rootgrp.createVariable("yd","f4",("t","x","y"),zlib=True)
ncyd.units = "m"
ncyd.standard_name = "y coordinate"
ncyd.grid_mapping = "crs"

ncgrad = rootgrp.createVariable("grad","f4",("x","y"),zlib=True)
ncgrad.units = "K"
ncgrad.standard_name = "Gradient correction"

# --- Write data
nctemp[:] = img_list
nctimes[:] = time_list
ncroll[:] = roll_list
ncpitch[:] = pitch_list
ncalti[:] = alti_list
nciix[:] = img_ix_list
nclon[:] = lon_list
nclat[:] = lat_list
ncgrad[:] = gradient_array.data
ncxd[:] = xd_list
ncyd[:] = yd_list
nchead[:] = mod_head_list
ncmss[:] = mss_list

rootgrp.close()
print('Done!')