"""
    A Python module for radar simulation

    ----------
    RadarSimPy - A Radar Simulator Built with Python
    Copyright (C) 2018 - PRESENT  Zhengyu Peng
    E-mail: zpeng.me@gmail.com
    Website: https://zpeng.me

    `                      `
    -:.                  -#:
    -//:.              -###:
    -////:.          -#####:
    -/:.://:.      -###++##:
    ..   `://:-  -###+. :##:
           `:/+####+.   :##:
    .::::::::/+###.     :##:
    .////-----+##:    `:###:
     `-//:.   :##:  `:###/.
       `-//:. :##:`:###/.
         `-//:+######/.
           `-/+####/.
             `+##+.
              :##:
              :##:
              :##:
              :##:
              :##:
               .+:

"""


from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import simc
from radarsimpy.simulator import simpy
from scipy import signal
import radarsimpy.processing as proc
import numpy as np
import numpy.testing as npt
import plotly.graph_objects as go
import plotly.io as pio

def test_arbitrary_waveform_cpp():
    tx_channel = dict(
        location=(0, 0, 0),
    )

    freq_nonlinear = np.array([
        2.40750000e+10, 2.40760901e+10, 2.40771786e+10, 2.40782654e+10,
        2.40793506e+10, 2.40804341e+10, 2.40815161e+10, 2.40825964e+10,
        2.40836750e+10, 2.40847521e+10, 2.40858275e+10, 2.40869012e+10,
        2.40879734e+10, 2.40890439e+10, 2.40901127e+10, 2.40911800e+10,
        2.40922456e+10, 2.40933096e+10, 2.40943719e+10, 2.40954326e+10,
        2.40964917e+10, 2.40975491e+10, 2.40986049e+10, 2.40996591e+10,
        2.41007117e+10, 2.41017626e+10, 2.41028119e+10, 2.41038595e+10,
        2.41049055e+10, 2.41059499e+10, 2.41069927e+10, 2.41080338e+10,
        2.41090733e+10, 2.41101111e+10, 2.41111473e+10, 2.41121819e+10,
        2.41132149e+10, 2.41142462e+10, 2.41152759e+10, 2.41163039e+10,
        2.41173304e+10, 2.41183552e+10, 2.41193783e+10, 2.41203999e+10,
        2.41214198e+10, 2.41224380e+10, 2.41234546e+10, 2.41244696e+10,
        2.41254830e+10, 2.41264947e+10, 2.41275048e+10, 2.41285133e+10,
        2.41295202e+10, 2.41305254e+10, 2.41315289e+10, 2.41325309e+10,
        2.41335312e+10, 2.41345298e+10, 2.41355269e+10, 2.41365223e+10,
        2.41375161e+10, 2.41385082e+10, 2.41394987e+10, 2.41404876e+10,
        2.41414748e+10, 2.41424605e+10, 2.41434444e+10, 2.41444268e+10,
        2.41454075e+10, 2.41463866e+10, 2.41473640e+10, 2.41483399e+10,
        2.41493140e+10, 2.41502866e+10, 2.41512575e+10, 2.41522268e+10,
        2.41531945e+10, 2.41541605e+10, 2.41551249e+10, 2.41560876e+10,
        2.41570488e+10, 2.41580083e+10, 2.41589661e+10, 2.41599224e+10,
        2.41608770e+10, 2.41618299e+10, 2.41627812e+10, 2.41637309e+10,
        2.41646790e+10, 2.41656254e+10, 2.41665702e+10, 2.41675134e+10,
        2.41684550e+10, 2.41693949e+10, 2.41703331e+10, 2.41712698e+10,
        2.41722048e+10, 2.41731381e+10, 2.41740699e+10, 2.41750000e+10])

    tx_nonlinear = Transmitter(f=freq_nonlinear,
                               t=np.linspace(0, 80e-6, 100),
                               tx_power=40,
                               prp=100e-6,
                               pulses=1,
                               channels=[tx_channel])

    rx_channel = dict(
        location=(0, 0, 0),
    )

    rx = Receiver(fs=2e6,
                  noise_figure=12,
                  rf_gain=20,
                  load_resistor=500,
                  baseband_gain=30,
                  channels=[rx_channel])

    radar_nonlinear = Radar(transmitter=tx_nonlinear, receiver=rx)

    target_1 = dict(location=(200, 0, 0), speed=(-5, 0, 0), rcs=20, phase=0)
    target_2 = dict(location=(95, 20, 0), speed=(-50, 0, 0), rcs=15, phase=0)
    target_3 = dict(location=(30, -5, 0), speed=(-22, 0, 0), rcs=5, phase=0)

    targets = [target_1, target_2, target_3]

    data_nonlinear = simc(radar_nonlinear, targets, noise=False)
    time_matrix_nonlinear = data_nonlinear['timestamp']
    data_matrix_nonlinear = data_nonlinear['baseband']

    range_window = signal.chebwin(radar_nonlinear.samples_per_pulse, at=60)

    range_profile_nonlinear = proc.range_fft(
        data_matrix_nonlinear[:, :, :], range_window)

    range_profile = 20 * np.log10(np.abs(range_profile_nonlinear[0, 0, :]))

    npt.assert_allclose(
        range_profile,
        np.array([
            -62.07849925, -62.20544263, -62.3609653, -62.52068991,
            -62.82363417, -63.09356487, -63.47789793, -64.18130834,
            -64.60987307, -66.02836818, -67.13391398, -70.23615273,
            -72.45375246, -70.23223916, -62.44448776, -55.28323117,
            -44.90115332, -29.81956035, -14.49433127,  -2.98812246,
            1.51677569,  -2.58718742, -13.81816558, -29.03688553,
            -44.14889749, -54.00768546, -60.09270676, -65.05351556,
            -69.6472799, -71.800232, -69.62770834, -68.94634515,
            -67.23959839, -66.4684642, -65.84670853, -65.18528611,
            -65.16536654, -64.42158345, -64.56269682, -64.21940498,
            -64.13403445, -63.96734378, -63.89617219, -63.92413024,
            -63.67429634, -63.79338911, -63.63722851, -63.5906069,
            -63.52304426, -63.44301377, -63.25882984, -62.97776338,
            -62.81673071, -62.29870445, -61.51990969, -60.17178873,
            -57.42435564, -52.11263294, -44.68573101, -36.52549748,
            -28.62836867, -21.75108895, -16.49123683, -13.25631776,
            -11.95225453, -12.47511286, -14.90360124, -19.33855372,
            -25.52285679, -33.04353452, -41.417082, -49.96591448,
            -57.38237831, -61.74431814, -63.67970419, -64.67236933,
            -65.60568643, -66.18370905, -67.04782568, -67.57896027,
            -67.45416173, -70.47643695, -68.39915052, -70.42200467,
            -70.77337434, -71.47014485, -72.1479753, -73.10817846,
            -73.42157313, -74.76747679, -74.98488992, -76.60993838,
            -76.86852787, -78.43480714, -78.86280941, -79.6829764,
            -81.03839696, -81.15559215, -81.15908901, -80.74672171,
            -80.91295929, -79.74260663, -78.73328366, -78.542396,
            -77.19543595, -76.88676823, -75.74101536, -75.42609875,
            -74.38697279, -74.391502, -73.31543608, -73.19368319,
            -72.73405164, -72.41793273, -72.0656123, -72.21822767,
            -71.98010131, -72.20412219, -73.42387124, -75.39698377,
            -83.62495715, -70.22757179, -60.84535869, -53.54270958,
            -47.15529457, -41.60347116, -36.94486216, -33.19081623,
            -30.1734301, -27.56615969, -25.41211038, -24.03246716,
            -23.01086849, -22.46216564, -22.46245021, -22.84614203,
            -23.71823433, -25.22794965, -26.95956288, -29.39757125,
            -32.74418113, -36.6561175, -40.81409585, -45.07456568,
            -49.43015643, -53.49715924, -56.91404228, -59.09205822,
            -60.32502533, -60.82214102, -61.18274026, -61.31916065,
            -61.41273532, -61.57555024, -61.63466541, -61.64666817,
            -61.73058582, -61.85520663, -61.81663542, -62.01865996]),
        rtol=3)


def test_arbitrary_waveform_py():
    tx_channel = dict(
        location=(0, 0, 0),
    )

    freq_nonlinear = np.array([
        2.40750000e+10, 2.40760901e+10, 2.40771786e+10, 2.40782654e+10,
        2.40793506e+10, 2.40804341e+10, 2.40815161e+10, 2.40825964e+10,
        2.40836750e+10, 2.40847521e+10, 2.40858275e+10, 2.40869012e+10,
        2.40879734e+10, 2.40890439e+10, 2.40901127e+10, 2.40911800e+10,
        2.40922456e+10, 2.40933096e+10, 2.40943719e+10, 2.40954326e+10,
        2.40964917e+10, 2.40975491e+10, 2.40986049e+10, 2.40996591e+10,
        2.41007117e+10, 2.41017626e+10, 2.41028119e+10, 2.41038595e+10,
        2.41049055e+10, 2.41059499e+10, 2.41069927e+10, 2.41080338e+10,
        2.41090733e+10, 2.41101111e+10, 2.41111473e+10, 2.41121819e+10,
        2.41132149e+10, 2.41142462e+10, 2.41152759e+10, 2.41163039e+10,
        2.41173304e+10, 2.41183552e+10, 2.41193783e+10, 2.41203999e+10,
        2.41214198e+10, 2.41224380e+10, 2.41234546e+10, 2.41244696e+10,
        2.41254830e+10, 2.41264947e+10, 2.41275048e+10, 2.41285133e+10,
        2.41295202e+10, 2.41305254e+10, 2.41315289e+10, 2.41325309e+10,
        2.41335312e+10, 2.41345298e+10, 2.41355269e+10, 2.41365223e+10,
        2.41375161e+10, 2.41385082e+10, 2.41394987e+10, 2.41404876e+10,
        2.41414748e+10, 2.41424605e+10, 2.41434444e+10, 2.41444268e+10,
        2.41454075e+10, 2.41463866e+10, 2.41473640e+10, 2.41483399e+10,
        2.41493140e+10, 2.41502866e+10, 2.41512575e+10, 2.41522268e+10,
        2.41531945e+10, 2.41541605e+10, 2.41551249e+10, 2.41560876e+10,
        2.41570488e+10, 2.41580083e+10, 2.41589661e+10, 2.41599224e+10,
        2.41608770e+10, 2.41618299e+10, 2.41627812e+10, 2.41637309e+10,
        2.41646790e+10, 2.41656254e+10, 2.41665702e+10, 2.41675134e+10,
        2.41684550e+10, 2.41693949e+10, 2.41703331e+10, 2.41712698e+10,
        2.41722048e+10, 2.41731381e+10, 2.41740699e+10, 2.41750000e+10])

    tx_nonlinear = Transmitter(f=freq_nonlinear,
                               t=np.linspace(0, 80e-6, 100),
                               tx_power=40,
                               prp=100e-6,
                               pulses=1,
                               channels=[tx_channel])

    rx_channel = dict(
        location=(0, 0, 0),
    )

    rx = Receiver(fs=2e6,
                  noise_figure=12,
                  rf_gain=20,
                  load_resistor=500,
                  baseband_gain=30,
                  channels=[rx_channel])

    radar_nonlinear = Radar(transmitter=tx_nonlinear, receiver=rx)

    target_1 = dict(location=(200, 0, 0), speed=(-5, 0, 0), rcs=20, phase=0)
    target_2 = dict(location=(95, 20, 0), speed=(-50, 0, 0), rcs=15, phase=0)
    target_3 = dict(location=(30, -5, 0), speed=(-22, 0, 0), rcs=5, phase=0)

    targets = [target_1, target_2, target_3]

    data_nonlinear = simpy(radar_nonlinear, targets, noise=False)
    time_matrix_nonlinear = data_nonlinear['timestamp']
    data_matrix_nonlinear = data_nonlinear['baseband']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_matrix_nonlinear[0, 0, :] * 1e6,
        y=np.real(data_matrix_nonlinear[0, 0, :]),
        name='I nonlinear',
    ))
    fig.add_trace(go.Scatter(
        x=time_matrix_nonlinear[0, 0, :] * 1e6,
        y=np.imag(data_matrix_nonlinear[0, 0, :]),
        name='Q nonlinear',
    ))
    # fig.add_trace(go.Scatter(
    #     x=time_matrix_linear[0, 0, :] * 1e6,
    #     y=np.real(data_matrix_linear[0, 0, :]),
    #     name='I linear',
    # ))
    # fig.add_trace(go.Scatter(
    #     x=time_matrix_linear[0, 0, :] * 1e6,
    #     y=np.imag(data_matrix_linear[0, 0, :]),
    #     name='Q linear',
    # ))

    fig.update_layout(
        template="seaborn",
        title='I/Q Beat Signals for the First Chirp',
        yaxis=dict(title='Amplitude (V)'),
        xaxis=dict(title='Time (µs)'),
    )

    # # fig.show()
    # display(SVG(fig.to_image(format='svg', scale=1)))
    # fig.update_layout(
    # template="seaborn",
    # title='I/Q Beat Signals for the First Chirp',
    # yaxis=dict(title='Amplitude (V)'),
    # xaxis=dict(title='Time (µs)'),
    # )

    # Save the figure as an SVG file
    pio.write_image(fig, 'figure.jpg', format='jpeg')
    print("saved")
    exit()






    range_window = signal.chebwin(radar_nonlinear.samples_per_pulse, at=60)

    range_profile_nonlinear = proc.range_fft(
        data_matrix_nonlinear[:, :, :], range_window)

    range_profile = 20 * np.log10(np.abs(range_profile_nonlinear[0, 0, :]))

    npt.assert_almost_equal(
        range_profile,
        np.array(
            [-60.3854425, -60.3835321, -60.09462924, -60.2359633,
             -60.0191639, -60.02067137, -59.89972158, -59.89701825,
             -59.84025448, -59.71529897, -59.64484282, -59.33458391,
             -58.73596819, -57.55811103, -55.30229423, -51.35907488,
             -43.47970724, -29.5766306, -14.49977084,  -2.99988942,
             1.52416371,  -2.59656805, -13.82018999, -28.80086609,
             -42.78589954, -50.58994683, -54.18924798, -56.21778977,
             -57.08117836, -57.55886187, -57.46158453, -57.52080834,
             -57.26358112, -57.16190049, -56.98434465, -56.85050682,
             -56.75935465, -56.48944871, -56.56819468, -56.34126641,
             -56.28963278, -56.15360848, -56.13103682, -56.03965947,
             -55.95667655, -55.87515271, -55.83736868, -55.73116961,
             -55.69886519, -55.61802082, -55.49032017, -55.40519097,
             -55.28840864, -55.10989003, -54.83571156, -54.35966784,
             -53.20583635, -49.82199387, -43.51169091, -35.92320848,
             -28.47172029, -21.82691248, -16.5328179, -13.21451633,
             -11.98708645, -12.43614681, -14.94696709, -19.34863399,
             -25.34832478, -32.76409038, -41.33327539, -50.29151853,
             -56.10811188, -56.73887927, -56.6510287, -56.50144903,
             -56.59057946, -56.63037637, -56.77600242, -56.8627721,
             -56.74017218, -57.39538901, -56.91829861, -57.33762536,
             -57.39332694, -57.48577212, -57.61169936, -57.75289106,
             -57.79210981, -57.99696083, -58.0175184, -58.23560847,
             -58.29088949, -58.46015347, -58.55684946, -58.66537296,
             -58.87462662, -58.944713, -59.1128894, -59.24959267,
             -59.39891224, -59.52861538, -59.69827673, -59.84756085,
             -60.01245265, -60.14283615, -60.32507466, -60.45395528,
             -60.68832391, -60.72046445, -60.99844953, -61.08364852,
             -61.22398836, -61.3526184, -61.46836298, -61.49808057,
             -61.55868003, -61.54869827, -61.28424207, -60.47508082,
             -59.71321516, -58.0097228, -55.27211389, -51.81951029,
             -47.50661986, -42.55450953, -37.57629517, -33.21211641,
             -29.89363079, -27.56358757, -25.57026272, -23.90742247,
             -23.07524128, -22.43586594, -22.48741444, -22.78794319,
             -23.83508673, -25.07435076, -26.98001904, -29.64445511,
             -32.66916881, -36.08744949, -40.18627128, -45.03906152,
             -50.63444982, -56.45071382, -61.33570758, -62.84100215,
             -62.70945956, -62.19637253, -61.87082861, -61.59453483,
             -61.35521129, -61.31564989, -60.98556589, -61.05573137,
             -60.7997435, -60.72459184, -60.56678135, -60.51839772]),
        decimal=1)

if(__name__ == "__main__"):
    test_arbitrary_waveform_py()