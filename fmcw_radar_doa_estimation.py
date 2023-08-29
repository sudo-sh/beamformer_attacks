import radarsimpy
import numpy as np
from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import simc

from scipy import signal
import radarsimpy.processing as proc


import plotly.graph_objects as go
import plotly.io as pio


'''
Observations

1. The CUDA library will not work on Ubuntu 20 as It neeeds GLIBC_2.35 which
is a dependency of librardarsimc.so

2. Num pulses in the FMCW radar (transmitter) is the critical block for the simulator


'''

    


def main():

    #lambda = c / f_c, where f_c is the center frequency of the 
    wavelength = 3e8/60.5e9

    N_tx = 2
    N_rx = 64

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
    '''
    tx = Transmitter(f=[61e9, 60e9], f_bot, f_top
                     t=[0, 16e-6], time of each pulse t[0] - t[1]
                     tx_power=15, dbm
                     prp=40e-6, pulse repetition periods
                     pulses=512, total number of pulses
                     channels=channels)                     
    '''
    tx = Transmitter(f=[61e9, 60e9],
                     t=[0, 16e-6],
                     tx_power=15,
                     prp=40e-6,
                     pulses=50,
                     channels=channels)

    

    #setup the Receiver
    channels = []
    for idx in range(0, N_rx):
        channels.append(
            dict(
                location=(0, wavelength/2*idx-(N_rx-1) * wavelength/4, 0),
            ))
    '''
    rx = Receiver(fs=20e6, Sampling rate
                  noise_figure=8,
                  rf_gain=20,
                  load_resistor=500, load resistor to convert power to voltage ohm
                  baseband_gain=30, db
                  channels=channels)
    Antennas are on the Y axis
    
    '''

                  
    rx = Receiver(fs=20e6,
                  noise_figure=8,
                  rf_gain=20,
                  load_resistor=500,
                  baseband_gain=30,
                  channels=channels)

    radar = Radar(transmitter=tx, receiver=rx)

    #Create Targets
    true_theta = [-5, -4, 45]

    target_1 = dict(location=(40*np.cos(np.radians(true_theta[0])), 40*np.sin(np.radians(true_theta[0])),
                    0), speed=(0, 0, 0), rcs=10, phase=0)
    target_2 = dict(location=(40*np.cos(np.radians(true_theta[1])), 40*np.sin(np.radians(true_theta[1])),
                    0), speed=(0, 0, 0), rcs=10, phase=0)
    target_3 = dict(location=(40*np.cos(np.radians(true_theta[2])), 40*np.sin(np.radians(true_theta[2])),
                    0), speed=(0, -40, 0), rcs=10, phase=0)

    targets = [target_1, target_2, target_3]

    ''''
    Simulator the baseband signals
    The output baseband data is 3D matrix
    [channels, pulses, ADC samples]
    '''

    print("Starting Simulator")
    bb_data = simc(radar, targets, noise=True)
    time_matrix = bb_data['timestamp']
    baseband = bb_data['baseband']
    print("End simulator")
    print("baseband signal",baseband.shape)
    # print(baseband[0][0][0])
    #The simulator returns the matrix of IQ channel data

    #Radar-Doppler Processing
    '''
    Create the chebysev window to taper the edges of a signal
    Tapering the signal help reduce spectral leakage and improves the
    frequency resolution when applying the Fourier Transwer
    at = Attenuation in dm
    '''
    range_window = signal.chebwin(radar.samples_per_pulse, at=80)
    doppler_window = signal.chebwin(radar.transmitter.pulses, at=60)

    range_doppler = proc.range_doppler_fft(
        baseband, rwin=range_window, dwin=doppler_window)
    print(range_doppler.shape)
    # print(range_doppler[0][0][0])
    '''
    processing range_doppler_fft
    takes the baseband data [channels, pulses, adc_samples]
    and returns the 3D array of range-Doppler map
    [channels, Doppler, range]

    '''

    '''
    Range Doppler average map
    A 3D array of range-Doppler map, ``[channels, Doppler, range]``
    '''
    max_range = (3e8 * radar.receiver.fs *
             radar.transmitter.pulse_length /
             radar.transmitter.bandwidth / 2)

    # print(radar.samples_per_pulse)
    range_axis = np.flip(np.linspace(
        0, max_range, radar.samples_per_pulse, endpoint=False))

    unambiguous_speed = 3e8 / radar.transmitter.prp[0] / \
        radar.transmitter.fc_vect[0] / 2
    
    """  
    Possbile variation in speed in the radial direction -unambiguous_speed/2 to +unambiguous_speed/2
    """

    doppler_axis = np.linspace(
        -unambiguous_speed, 0, radar.transmitter.pulses, endpoint=False)
    # print(radar.transmitter.pulses)

    fig = go.Figure()
    fig.add_trace(go.Surface(x=range_axis, y=doppler_axis, z=20 *
                  np.log10(np.mean(np.abs(range_doppler), axis=0)), colorscale='Rainbow'))

    fig.update_layout(
        title='Range Doppler',
        height=600,
        scene=dict(
            xaxis=dict(title='Range (m)'),
            yaxis=dict(title='Velocity (m/s)'),
            zaxis=dict(title='Amplitude (dB)'),
        ),
        margin=dict(l=0, r=0, b=60, t=100),
        legend=dict(orientation='h'),
    )

    # fig.show()
    # Image(fig.to_image(format="jpg", scale=2))
    pio.write_image(fig, 'figure.jpg', format='jpeg')
    t = range_doppler[:, 0, :]
    print(t.shape)

    det_idx = [np.argmax(np.mean(np.abs(range_doppler[:, 0, :]), axis=0))]
    print(det_idx)



if(__name__ == "__main__"):
    main()