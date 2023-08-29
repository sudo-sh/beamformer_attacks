import radarsimpy
import numpy as np
from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import simc, simpy

from scipy import signal
import radarsimpy.processing as proc
from radarsimpy.processing import doa_music, doa_root_music, doa_esprit

from scipy import linalg
from scipy import fft

import plotly.graph_objects as go
import plotly.io as pio

import json

from targets import *
from plot import *

import logging
import datetime
import time
# import tqdm
from tqdm import tqdm

c= 3e8


############Setup Logging################
path = "logs/"
# Get the current date and time
current_datetime = datetime.datetime.now()
# Generate the log file name with the current date and time
log_filename = current_datetime.strftime(path +"/logfile_%Y-%m-%d_%H-%M-%S.log")
# Configure logging format and level, and use the generated log filename
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=log_filename,
    filemode='a'
)



def read_config(PATH_to_config = "config.json"):
    # Read the contents of the config.json file
    with open(PATH_to_config, 'r') as f:
        config_data = json.load(f)
    return config_data


def setup_tx(config_data):
    """
    Read about the antenna placement for the TX and RX,
    Currently I can support only 2 TX and any number of RX
    """

    N_rx = config_data["rx_params"]["n_rx"]
    wavelength = c /((config_data["tx_params"]["fc_top"] + config_data["tx_params"]["fc_bot"])/2)

    #Setup Transmitter
    channels = []
    #location is of the form of x, y, z
    #place the two transmitters on the extreme ends of the receivers 
    channels.append(
        dict(
            location=(0, -N_rx/2*wavelength/2, 0),
        ))

    channels.append(
        dict(
            location=(0, wavelength*N_rx/2-N_rx/2*wavelength/2, 0),
        ))
    
    tx = Transmitter(f=[config_data["tx_params"]["fc_top"], config_data["tx_params"]["fc_bot"]],
                     t=[0, config_data["tx_params"]["t_pulse"]],
                     tx_power=config_data["tx_params"]["tx_power"],
                     prp=config_data["tx_params"]["prp"],
                     pulses=config_data["tx_params"]["n_pulses"],
                     channels=channels)
    return tx



def setup_rx(config_data):

    N_rx = config_data["rx_params"]["n_rx"]
    wavelength = c /((config_data["tx_params"]["fc_top"] + config_data["tx_params"]["fc_bot"])/2)

    #setup the Receiver
    channels = []
    for idx in range(0, N_rx):
        channels.append(
            dict(
                location=(0, wavelength/2*idx-(N_rx-1) * wavelength/4, 0),
            ))
    
    rx = Receiver(fs=config_data["rx_params"]["f_samp"],
              noise_figure=config_data["rx_params"]["noise_figure"],
              rf_gain=config_data["rx_params"]["rf_gain"],
              load_resistor=config_data["rx_params"]["load_resistor"],
              baseband_gain=config_data["rx_params"]["baseband_gain"],
              channels=channels)

    return rx

def range_doppler_proc(radar, baseband):

    range_window = signal.windows.chebwin(radar.samples_per_pulse, at=80)
    doppler_window = signal.windows.chebwin(radar.transmitter.pulses, at=80)

    range_doppler = proc.range_doppler_fft(
        baseband, rwin=range_window, dwin=doppler_window)
    
    return range_doppler
   

def find_unique_within_tolerance(arr, tol):
    unique_values = np.unique(arr)
    unique_within_tol = []

    for value in unique_values:
        # Check if the value is not already present in 'unique_within_tol' within the tolerance 'tol'
        if not any(np.isclose(unique_within_tol, value, atol=tol)):
            unique_within_tol.append(value)

    return np.array(unique_within_tol)

def return_unique_range_doppler(radar, config_data, det_ranges, det_vel_wo_direction):
    max_range = (c * radar.receiver.fs *
             radar.transmitter.pulse_length /
             radar.transmitter.bandwidth / 2)
    max_speed = c /radar.transmitter.prp[0] / \
        radar.transmitter.fc_vect[0] / 2
    
    unique_det_range = find_unique_within_tolerance(det_ranges, tol = config_data["hyper_params"]["range_tol"]*max_range/radar.transmitter.pulses )
    # print("radial_distance",unique_det_range)

    unique_det_indices = np.ones((len(unique_det_range)), dtype = int)
    for i in range(0, len(unique_det_range)):
        unique_det_indices[i]= int(np.median(np.where(det_ranges == unique_det_range[i])[0]))
   
    unique_det_dop_wo_dir = det_vel_wo_direction[unique_det_indices]
   
    unique_det_dop_w_dir = np.zeros((len(unique_det_dop_wo_dir)))
    for i in range(0, len(unique_det_dop_wo_dir)):
        if(unique_det_dop_wo_dir[i] < -0.5*max_speed):
            unique_det_dop_w_dir[i] = (unique_det_dop_wo_dir[i] + max_speed)
        else:
            unique_det_dop_w_dir[i] = (unique_det_dop_wo_dir[i])

    return unique_det_range, unique_det_dop_w_dir, unique_det_dop_wo_dir

def peakdet_and_convmat(config_data, radar, range_doppler):

    N_rx = config_data["rx_params"]["n_rx"]
    N_tx = config_data["tx_params"]["n_tx"]

    '''
    Peak detection algorithm
    Range doppler has the dimensions
    [channels, Doppler, range]
    '''
    # peaks, _ = signal.find_peaks(range_doppler, height=config_data["hyper_params"]["peak_thresh"])
    # print(peaks)
    # exit()
    range_doppler_map = 20*np.log10(np.mean(np.abs(range_doppler), axis=0))
    doppler_ind, range_ind = np.where(range_doppler_map > config_data["hyper_params"]["peak_thresh"] )
    
    """
    Map to range velocity and then we get back
    """
  
    max_range = (c * radar.receiver.fs *
             radar.transmitter.pulse_length /
             radar.transmitter.bandwidth / 2)
    max_speed = c /radar.transmitter.prp[0] / \
        radar.transmitter.fc_vect[0] / 2
    
    det_ranges = -1*np.interp(range_ind, [0, radar.samples_per_pulse],[-1*max_range, 0])
    det_vel_wo_direction = np.interp(doppler_ind, [0, radar.transmitter.pulses], [-1*max_speed, 0])
    # print(det_ranges)
    # print(det_vel_wo_direction)
    
    unique_det_range, unique_det_dop_w_dir, unique_det_dop_wo_dir = return_unique_range_doppler(radar, config_data, det_ranges, det_vel_wo_direction)
    # print("radial_distance",unique_det_range)
    # print("velocity", unique_det_dop_w_dir)

    det_ranges_ind = range_ind
    det_vel_dir_ind = doppler_ind
    # print(det_ranges_ind)
    # print(det_vel_dir_ind)
    # det_ranges_ind = np.interp(-1*unique_det_range, [-1*max_range, 0], [0, radar.samples_per_pulse])
    # det_vel_dir_ind = np.interp(unique_det_dop_wo_dir, [-1*max_speed, 0], [0, radar.transmitter.pulses])
    # print(range_ind, doppler_ind)
    # exit()
    # det_ranges_ind = range_ind

    '''
    The dimension of range_ind and doppler_ind is the same as the number of target.
    We will iterate through all the targets create the covariance matrix based on the 
    Ns observation snapshots
    '''

    bv_all = []

    for i in range(0, len(det_ranges_ind)):
        bv =range_doppler[:,int(det_vel_dir_ind[i]),int(det_ranges_ind[i])]
        bv = bv/linalg.norm(bv)
        bv_all.append(bv)
    bv_all_np = np.array(bv_all)
    bv_all_music = bv_all_np.T
    cov_mat = np.cov(bv_all_music.conjugate())

    # scan_angle = np.arange(-90, 90, 0.1)
    # music_doa, music_idx, ps_db = doa_music(cov_mat, config_data["target"]["num_target"], scanangles=scan_angle)

    return cov_mat, unique_det_range, unique_det_dop_w_dir
   
def run_sim(config_data, logging):
    # start_time = time.time()

    radar = Radar(transmitter = setup_tx(config_data), receiver = setup_rx(config_data))

    num_epoch = config_data["run_params"]["run_itr"]
    # print(num_epoch)

    for i in tqdm(range(num_epoch)):

        range_arr, angle_arr, speed_arr, targets = get_targets(radar, config_data, logging)
        # print(ranges, angles, speed)
        # exit()
        # print("############Starting Simulator##############")
        # bb_data = simpy(radar, targets, noise=True)
        bb_data = simc(radar, targets, noise=True)
        time_matrix = bb_data['timestamp']
        baseband = bb_data['baseband']
        # print("###########End simulator###################")

        range_doppler = range_doppler_proc(radar, baseband)
        # plot_range_doppler(radar, range_doppler)

        #########################Different DoA Estimators#################################################

        #Find the beam vector of the peak and create covariance matrix for DoA
        # covmat = peakdet_and_convmat(config_data, radar, range_doppler)
        cov_mat, unique_det_range, unique_det_dop_w_dir = peakdet_and_convmat(config_data, radar, range_doppler)
        # plot_peaks(range_doppler)

        logging.info("radial_distance" + str(unique_det_range))
        logging.info("velocity"+ str(unique_det_dop_w_dir))

        scan_angle = np.arange(-90, 90, 0.1)
        music_doa, music_idx, ps_db = doa_music(cov_mat, config_data["target"]["num_target"], scanangles=scan_angle)

        logging.info('DoAs from MUSIC:'+ str(music_doa) + 'degrees')
    

        # rootmusic_doa = doa_root_music(cov_mat, config_data["target"]["num_target"])
        # logging.info('DoAs from root MUSIC:'+ str(rootmusic_doa) + 'degrees')

        # esprit_doa = doa_esprit(cov_mat, config_data["target"]["num_target"])
        # logging.info('DoAs from ESPRIT:'+ str(esprit_doa) + 'degrees')


        """
        Save the results:
        We would be saving the the target info
        range doppler np array --> It is very big so don't save now
        and covariance matrix np array
        """

        # np.savez(config_data["run_params"]["save_path"]+"/epoch_test.npz", arr1 = range_arr, arr2 = speed_arr,
                #  arr3 = angle_arr, arr4 = range_doppler, arr5 = cov_mat)
        # print(range_arr, angle_arr, speed_arr)
        # print(cov_mat.shape)
        np.savez(config_data["run_params"]["save_path"]+"/epoch_"+str(i)+".npz", range = range_arr, speed = speed_arr,
                 angle = angle_arr, cov_mat = cov_mat)

        # end_time = time.time()
        # print("Execution Time", end_time - start_time )
   
 
def main():
    
    config_data = read_config()

    #dump configs to logging
    logging.info(config_data)
    run_sim(config_data, logging)

if(__name__ == "__main__"):
    main()