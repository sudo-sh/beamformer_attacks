from radarsimpy import Radar, Transmitter, Receiver

import numpy as np

from scipy import linalg
from scipy import fft


import plotly.graph_objects as go
import plotly.io as pio
    

def plot_range_doppler(radar, range_doppler):
    max_range = (3e8 * radar.receiver.fs *
             radar.transmitter.pulse_length /
             radar.transmitter.bandwidth / 2)

    # print(radar.samples_per_pulse)
    range_axis = np.flip(np.linspace(
        0, max_range, radar.samples_per_pulse, endpoint=False))
    # range_axis = np.linspace(
    #     0, max_range, radar.samples_per_pulse, endpoint=False)

    unambiguous_speed = 3e8 / radar.transmitter.prp[0] / \
        radar.transmitter.fc_vect[0] / 2
    
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



def plot_peaks(range_doppler):

    det_idx = [np.argmax(np.mean(np.abs(range_doppler[:, 0, :]), axis=0))]

    bv = range_doppler[:, 0, det_idx[0]]
    bv = bv/linalg.norm(bv)

    fft_spec = 20 * np.log10(np.abs(fft.fftshift(fft.fft(bv.conjugate(), n=1024))))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arcsin(np.linspace(-1, 1, 1024, endpoint=False))/np.pi*180,
                             y=fft_spec,
                             name='FFT')
                  )

    fig.update_layout(
        title='FFT',
        yaxis=dict(title='Amplitude (dB)'),
        xaxis=dict(title='Angle (deg)'),
        margin=dict(l=10, r=10, b=10, t=40),
    )

    # fig.show()
    # Image(fig.to_image(format="jpg", scale=2))
    pio.write_image(fig, 'figure_fft_peaks.jpg', format='jpeg')
