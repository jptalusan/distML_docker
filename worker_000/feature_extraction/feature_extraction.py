import numpy as np
import pandas as pd
import math
from scipy import signal
from scipy.fftpack import fft, fftfreq
from scipy.stats import iqr
from scipy.stats import pearsonr
from scipy.stats import skew
from scipy.stats import kurtosis
from spectrum import *


def sliding_window(data, window, step_size):
    shape = (int(data.shape[-1] / window * window / step_size - 1), window)
    strides = (data.strides[-1] * step_size, data.strides[-1])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def median_filter(data, f_size=3):
    return signal.medfilt(data, f_size)


def butter_lowpass(cutoff, fs, order=3):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=3):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=20.0, fs=50.0, order=3):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return signal.lfilter(b, a, data)


def butter_highpass_filter(data, cutoff=20.0, fs=50.0, order=3):
    b, a = butter_highpass(cutoff, fs, order=order)
    return signal.lfilter(b, a, data)


def calc_jerk(data):
    t = np.diff(data, axis=0)
    return np.insert(t, 0, 0)


def magnitude(x, y, z):
    return np.linalg.norm([x, y, z])


def graph_plot(data):
    pd.DataFrame(data).plot()


def calc_fft_signal(data):
    shape = data.shape
    N = shape[0]  # FFTのサンプル数
    hanningWindow = np.hanning(N)  # ハニング窓
    fqy = np.abs(fft(hanningWindow * data))
    return fqy[0:int(shape[0] / 2)]


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x - min) / (max - min)
    return result


def zscore(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x - xmean) / xstd
    return zscore


def mad(data, axis=None):
    """
    Compute *Median Absolute Deviation* of an array along given axis.
    """
    med = np.median(data, axis=axis, keepdims=True)
    mad = np.median(np.absolute(data - med), axis=axis)  # MAD along given axis
    return mad


def sma(data_x, data_y, data_z):
    """
    Compute *Signal magnitude area*.
    """
    sum = 0
    # print(data_x)
    for i in range(len(data_x)):
        sum += (abs(data_x[i]) + abs(data_y[i]) + abs(data_z[i]))
    return sum / len(data_x)


def sma_mag(data_mag):
    """
    Compute *Signal magnitude area*.
    """
    sum = 0
    for i in range(len(data_mag)):
        sum += abs(data_mag[i])

    return sum / len(data_mag)


def energy(data):
    """
    Compute *Energy measure value*.
    """
    energy = np.sum(data ** 2) / len(data)
    return energy


def shannon_entropy(data):
    ent = 0.0
    for freq in data:
        if freq > 0:
            ent = ent + freq * math.log(freq, 2)
    ent = -ent
    return ent


def arCoeff(data):
    """
    Compute *Autorregresion coefficients*.
    """
    AR, P, k = arburg(min_max(data), 4)
    float_AR = [numpy.real(n) for n in AR]
    #float_AR = [float(n) for n in AR]
    return float_AR


def calc_angle(v1, v2, acute=True):
    # v1 is your firsr vector
    # v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return abs(angle)
    else:
        return 2 * np.pi - angle


def rms(data):
    """
    Compute *root mean square* of an array.
    """
    rms = np.sqrt(np.mean(data ** 2))

    return rms


def time_xyz_feature_extraction(data_x, data_y, data_z):
    f_mean_x = data_x.mean()
    f_mean_y = data_y.mean()
    f_mean_z = data_z.mean()
    f_std_x = data_x.std()
    f_std_y = data_y.std()
    f_std_z = data_z.std()
    f_mad_x = mad(data_x)
    f_mad_y = mad(data_y)
    f_mad_z = mad(data_z)
    f_max_x = data_x.max()
    f_max_y = data_y.max()
    f_max_z = data_z.max()
    f_min_x = data_x.min()
    f_min_y = data_y.min()
    f_min_z = data_z.min()
    f_sma = sma(data_x, data_y, data_z)
    f_energy_x = energy(data_x)
    f_energy_y = energy(data_y)
    f_energy_z = energy(data_z)
    f_iqr_x = iqr(data_x)
    f_iqr_y = iqr(data_y)
    f_iqr_z = iqr(data_z)
    f_entropy_x = shannon_entropy(data_x)
    f_entropy_y = shannon_entropy(data_y)
    f_entropy_z = shannon_entropy(data_z)
    f_arCoeff_x = arCoeff(data_x)
    f_arCoeff_y = arCoeff(data_y)
    f_arCoeff_z = arCoeff(data_z)
    f_correlation_xy = pearsonr(data_x, data_y)[0]
    f_correlation_xz = pearsonr(data_x, data_z)[0]
    f_correlation_yz = pearsonr(data_y, data_z)[0]

    f_all = [f_mean_x, f_mean_y, f_mean_z, f_std_x, f_std_y, f_std_z, f_mad_x, f_mad_y, f_mad_z, \
             f_max_x, f_max_y, f_max_z, f_min_x, f_min_y, f_min_z, f_sma, f_energy_x, f_energy_y, f_energy_z, \
             f_iqr_x, f_iqr_y, f_iqr_z, f_entropy_x, f_entropy_y, f_entropy_z] + f_arCoeff_x + f_arCoeff_y + f_arCoeff_z \
            + [f_correlation_xy, f_correlation_xz, f_correlation_yz]
    # print(len(f_all))

    return np.array(f_all)


def time_mag_feature_extraction(data_mag):
    f_mean_mag = data_mag.mean()
    f_std_mag = data_mag.std()
    f_mad_mag = mad(data_mag)
    f_max_mag = data_mag.max()
    f_min_mag = data_mag.min()
    f_sma_mag = sma_mag(data_mag)
    f_energy_mag = energy(data_mag)
    f_iqr_mag = iqr(data_mag)
    f_entropy_mag = shannon_entropy(data_mag)
    f_arCoeff_mag = arCoeff(data_mag)

    f_all = [f_mean_mag, f_std_mag, f_mad_mag, \
             f_max_mag, f_min_mag, f_sma_mag, f_energy_mag, \
             f_iqr_mag, f_entropy_mag] + f_arCoeff_mag
    # print(len(f_all))

    return np.array(f_all)


def frequency_xyz_feature_extraction(data_x, data_y, data_z, fs=50.0):
    df_freqList = pd.DataFrame(fftfreq(int(2 * len(data_x)), d=1.0 / fs)[0:len(data_x)])

    f_mean_x = data_x.mean()
    f_mean_y = data_y.mean()
    f_mean_z = data_z.mean()
    f_std_x = data_x.std()
    f_std_y = data_y.std()
    f_std_z = data_z.std()
    f_mad_x = mad(data_x)
    f_mad_y = mad(data_y)
    f_mad_z = mad(data_z)
    f_max_x = data_x.max()
    f_max_y = data_y.max()
    f_max_z = data_z.max()
    f_min_x = data_x.min()
    f_min_y = data_y.min()
    f_min_z = data_z.min()
    f_sma = sma(data_x, data_y, data_z)
    f_energy_x = energy(data_x)
    f_energy_y = energy(data_y)
    f_energy_z = energy(data_z)
    f_iqr_x = iqr(data_x)
    f_iqr_y = iqr(data_y)
    f_iqr_z = iqr(data_z)
    f_entropy_x = shannon_entropy(data_x)
    f_entropy_y = shannon_entropy(data_y)
    f_entropy_z = shannon_entropy(data_z)
    f_maxinds_x = df_freqList[0][data_x.argmax()]
    f_maxinds_y = df_freqList[0][data_y.argmax()]
    f_maxinds_z = df_freqList[0][data_z.argmax()]
    f_meanFreq_x = np.average(df_freqList[0], weights=data_x)
    f_meanFreq_y = np.average(df_freqList[0], weights=data_y)
    f_meanFreq_z = np.average(df_freqList[0], weights=data_z)
    f_skewness_x = skew(data_x)
    f_skewness_y = skew(data_y)
    f_skewness_z = skew(data_z)
    f_kurtosis_x = kurtosis(data_x)
    f_kurtosis_y = kurtosis(data_y)
    f_kurtosis_z = kurtosis(data_z)

    f_bandsEnergy_x0 = np.sum(data_x.reshape(8, int(len(data_x) / 8))[0] ** 2) / len(data_x)
    f_bandsEnergy_x1 = np.sum(data_x.reshape(8, int(len(data_x) / 8))[1] ** 2) / len(data_x)
    f_bandsEnergy_x2 = np.sum(data_x.reshape(8, int(len(data_x) / 8))[2] ** 2) / len(data_x)
    f_bandsEnergy_x3 = np.sum(data_x.reshape(8, int(len(data_x) / 8))[3] ** 2) / len(data_x)
    f_bandsEnergy_x4 = np.sum(data_x.reshape(8, int(len(data_x) / 8))[4] ** 2) / len(data_x)
    f_bandsEnergy_x5 = np.sum(data_x.reshape(8, int(len(data_x) / 8))[5] ** 2) / len(data_x)
    f_bandsEnergy_x6 = np.sum(data_x.reshape(8, int(len(data_x) / 8))[6] ** 2) / len(data_x)
    f_bandsEnergy_x7 = np.sum(data_x.reshape(8, int(len(data_x) / 8))[7] ** 2) / len(data_x)
    f_bandsEnergy_x8 = np.sum(data_x.reshape(4, int(len(data_x) / 4))[0] ** 2) / len(data_x)
    f_bandsEnergy_x9 = np.sum(data_x.reshape(4, int(len(data_x) / 4))[1] ** 2) / len(data_x)
    f_bandsEnergy_x10 = np.sum(data_x.reshape(4, int(len(data_x) / 4))[2] ** 2) / len(data_x)
    f_bandsEnergy_x11 = np.sum(data_x.reshape(4, int(len(data_x) / 4))[3] ** 2) / len(data_x)
    f_bandsEnergy_x12 = np.sum(data_x.reshape(2, int(len(data_x) / 2))[0] ** 2) / len(data_x)
    f_bandsEnergy_x13 = np.sum(data_x.reshape(2, int(len(data_x) / 2))[1] ** 2) / len(data_x)

    f_bandsEnergy_y0 = np.sum(data_y.reshape(8, int(len(data_y) / 8))[0] ** 2) / len(data_y)
    f_bandsEnergy_y1 = np.sum(data_y.reshape(8, int(len(data_y) / 8))[1] ** 2) / len(data_y)
    f_bandsEnergy_y2 = np.sum(data_y.reshape(8, int(len(data_y) / 8))[2] ** 2) / len(data_y)
    f_bandsEnergy_y3 = np.sum(data_y.reshape(8, int(len(data_y) / 8))[3] ** 2) / len(data_y)
    f_bandsEnergy_y4 = np.sum(data_y.reshape(8, int(len(data_y) / 8))[4] ** 2) / len(data_y)
    f_bandsEnergy_y5 = np.sum(data_y.reshape(8, int(len(data_y) / 8))[5] ** 2) / len(data_y)
    f_bandsEnergy_y6 = np.sum(data_y.reshape(8, int(len(data_y) / 8))[6] ** 2) / len(data_y)
    f_bandsEnergy_y7 = np.sum(data_y.reshape(8, int(len(data_y) / 8))[7] ** 2) / len(data_y)
    f_bandsEnergy_y8 = np.sum(data_y.reshape(4, int(len(data_y) / 4))[0] ** 2) / len(data_y)
    f_bandsEnergy_y9 = np.sum(data_y.reshape(4, int(len(data_y) / 4))[1] ** 2) / len(data_y)
    f_bandsEnergy_y10 = np.sum(data_y.reshape(4, int(len(data_y) / 4))[2] ** 2) / len(data_y)
    f_bandsEnergy_y11 = np.sum(data_y.reshape(4, int(len(data_y) / 4))[3] ** 2) / len(data_y)
    f_bandsEnergy_y12 = np.sum(data_y.reshape(2, int(len(data_y) / 2))[0] ** 2) / len(data_y)
    f_bandsEnergy_y13 = np.sum(data_y.reshape(2, int(len(data_y) / 2))[1] ** 2) / len(data_y)

    f_bandsEnergy_z0 = np.sum(data_z.reshape(8, int(len(data_z) / 8))[0] ** 2) / len(data_z)
    f_bandsEnergy_z1 = np.sum(data_z.reshape(8, int(len(data_z) / 8))[1] ** 2) / len(data_z)
    f_bandsEnergy_z2 = np.sum(data_z.reshape(8, int(len(data_z) / 8))[2] ** 2) / len(data_z)
    f_bandsEnergy_z3 = np.sum(data_z.reshape(8, int(len(data_z) / 8))[3] ** 2) / len(data_z)
    f_bandsEnergy_z4 = np.sum(data_z.reshape(8, int(len(data_z) / 8))[4] ** 2) / len(data_z)
    f_bandsEnergy_z5 = np.sum(data_z.reshape(8, int(len(data_z) / 8))[5] ** 2) / len(data_z)
    f_bandsEnergy_z6 = np.sum(data_z.reshape(8, int(len(data_z) / 8))[6] ** 2) / len(data_z)
    f_bandsEnergy_z7 = np.sum(data_z.reshape(8, int(len(data_z) / 8))[7] ** 2) / len(data_z)
    f_bandsEnergy_z8 = np.sum(data_z.reshape(4, int(len(data_z) / 4))[0] ** 2) / len(data_z)
    f_bandsEnergy_z9 = np.sum(data_z.reshape(4, int(len(data_z) / 4))[1] ** 2) / len(data_z)
    f_bandsEnergy_z10 = np.sum(data_z.reshape(4, int(len(data_z) / 4))[2] ** 2) / len(data_z)
    f_bandsEnergy_z11 = np.sum(data_z.reshape(4, int(len(data_z) / 4))[3] ** 2) / len(data_z)
    f_bandsEnergy_z12 = np.sum(data_z.reshape(2, int(len(data_z) / 2))[0] ** 2) / len(data_z)
    f_bandsEnergy_z13 = np.sum(data_z.reshape(2, int(len(data_z) / 2))[1] ** 2) / len(data_z)

    f_all = [f_mean_x, f_mean_y, f_mean_z, f_std_x, f_std_y, f_std_z, f_mad_x, f_mad_y, f_mad_z, \
             f_max_x, f_max_y, f_max_z, f_min_x, f_min_y, f_min_z, f_sma, f_energy_x, f_energy_y, f_energy_z, \
             f_iqr_x, f_iqr_y, f_iqr_z, f_entropy_x, f_entropy_y, f_entropy_z, f_maxinds_x, f_maxinds_y, f_maxinds_z, \
             f_meanFreq_x, f_meanFreq_y, f_meanFreq_z, f_skewness_x, f_skewness_y, f_skewness_z, \
             f_kurtosis_x, f_kurtosis_y, f_kurtosis_z, f_bandsEnergy_x0, f_bandsEnergy_x1, f_bandsEnergy_x2, \
             f_bandsEnergy_x3, f_bandsEnergy_x4, f_bandsEnergy_x5, f_bandsEnergy_x6, f_bandsEnergy_x7, \
             f_bandsEnergy_x8, f_bandsEnergy_x9, f_bandsEnergy_x10, f_bandsEnergy_x11, f_bandsEnergy_x12,
             f_bandsEnergy_x13, \
             f_bandsEnergy_y0, f_bandsEnergy_y1, f_bandsEnergy_y2, \
             f_bandsEnergy_y3, f_bandsEnergy_y4, f_bandsEnergy_y5, f_bandsEnergy_y6, f_bandsEnergy_y7, \
             f_bandsEnergy_y8, f_bandsEnergy_y9, f_bandsEnergy_y10, f_bandsEnergy_y11, f_bandsEnergy_y12,
             f_bandsEnergy_y13, \
             f_bandsEnergy_z0, f_bandsEnergy_z1, f_bandsEnergy_z2, \
             f_bandsEnergy_z3, f_bandsEnergy_z4, f_bandsEnergy_z5, f_bandsEnergy_z6, f_bandsEnergy_z7, \
             f_bandsEnergy_z8, f_bandsEnergy_z9, f_bandsEnergy_z10, f_bandsEnergy_z11, f_bandsEnergy_z12,
             f_bandsEnergy_z13]

    # print(len(f_all))
    # print(f_all)

    return np.array(f_all)


def frequency_mag_feature_extraction(data_mag, fs=50.0):
    df_freqList = pd.DataFrame(fftfreq(int(2 * len(data_mag)), d=1.0 / fs)[0:len(data_mag)])

    f_mean_mag = data_mag.mean()

    f_std_mag = data_mag.std()

    f_mad_mag = mad(data_mag)

    f_max_mag = data_mag.max()

    f_min_mag = data_mag.min()

    f_sma_mag = sma_mag(data_mag)

    f_energy_mag = energy(data_mag)

    f_iqr_mag = iqr(data_mag)

    f_entropy_mag = shannon_entropy(data_mag)

    f_maxinds_mag = df_freqList[0][data_mag.argmax()]

    f_meanFreq_mag = np.average(df_freqList[0], weights=data_mag)

    f_skewness_mag = skew(data_mag)

    f_kurtosis_mag = kurtosis(data_mag)

    f_all = [f_mean_mag, f_std_mag, f_mad_mag, \
             f_max_mag, f_min_mag, f_sma_mag, f_energy_mag, \
             f_iqr_mag, f_entropy_mag, f_maxinds_mag, f_meanFreq_mag, f_skewness_mag, f_kurtosis_mag]

    # print(len(f_all))

    return np.array(f_all)


def angle_feature_extraction(bodyacc_x, bodyacc_y, bodyacc_z, \
                             bodyaccJerk_x, bodyaccJerk_y, bodyaccJerk_z, \
                             bodygyro_x, bodygyro_y, bodygyro_z, \
                             bodygyroJerk_x, bodygyroJerk_y, bodygyroJerk_z, \
                             gravity_x, gravity_y, gravity_z):
    tBodyAcc_Gravity = calc_angle([bodyacc_x.mean(), bodyacc_y.mean(), bodyacc_z.mean()], \
                                  [gravity_x.mean(), gravity_y.mean(), gravity_z.mean()])

    tBodyAccJerk_Gravity = calc_angle([bodyaccJerk_x.mean(), bodyaccJerk_y.mean(), bodyaccJerk_z.mean()], \
                                      [gravity_x.mean(), gravity_y.mean(), gravity_z.mean()])

    tBodyGyro_Gravity = calc_angle([bodygyro_x.mean(), bodygyro_y.mean(), bodygyro_z.mean()], \
                                   [gravity_x.mean(), gravity_y.mean(), gravity_z.mean()])

    tBodyGyroJerk_Gravity = calc_angle([bodygyroJerk_x.mean(), bodygyroJerk_y.mean(), bodygyroJerk_z.mean()], \
                                       [gravity_x.mean(), gravity_y.mean(), gravity_z.mean()])

    tBodyAcc_X = calc_angle([1, 0, 0], [gravity_x.mean(), gravity_y.mean(), gravity_z.mean()])
    tBodyAcc_Y = calc_angle([0, 1, 0], [gravity_x.mean(), gravity_y.mean(), gravity_z.mean()])
    tBodyAcc_Z = calc_angle([0, 0, 1], [gravity_x.mean(), gravity_y.mean(), gravity_z.mean()])

    f_all = [tBodyAcc_Gravity, tBodyAccJerk_Gravity, tBodyGyro_Gravity, tBodyGyroJerk_Gravity, tBodyAcc_X, tBodyAcc_Y,
             tBodyAcc_Z]

    # print(len(f_all))

    return np.array(f_all)


def additional_feature_extraction(data_x, data_y, data_z, data_mag, fs=50.0):
    f_rms_x = np.sqrt((data_x ** 2).mean())
    f_rms_y = np.sqrt((data_y ** 2).mean())
    f_rms_z = np.sqrt((data_z ** 2).mean())
    f_rms_mag = np.sqrt((data_mag ** 2).mean())

    f_ptp_x = data_x.ptp()
    f_ptp_y = data_y.ptp()
    f_ptp_z = data_z.ptp()
    f_ptp_mag = data_mag.ptp()

    f_pD_x0 = signal.welch(data_x, fs, nperseg=len(data_x))[1][0:int(len(data_x) / 8)].mean()
    f_pD_x1 = signal.welch(data_x, fs, nperseg=len(data_x))[1][int(len(data_x) / 8):int(len(data_x) / 8 * 2)].mean()
    f_pD_x2 = signal.welch(data_x, fs, nperseg=len(data_x))[1][int(len(data_x) / 8 * 2):int(len(data_x) / 8 * 3)].mean()
    f_pD_x3 = signal.welch(data_x, fs, nperseg=len(data_x))[1][int(len(data_x) / 8 * 3):int(len(data_x) / 2)].mean()

    f_pD_y0 = signal.welch(data_y, fs, nperseg=len(data_y))[1][0:int(len(data_y) / 8)].mean()
    f_pD_y1 = signal.welch(data_y, fs, nperseg=len(data_y))[1][int(len(data_y) / 8):int(len(data_y) / 8 * 2)].mean()
    f_pD_y2 = signal.welch(data_y, fs, nperseg=len(data_y))[1][int(len(data_y) / 8 * 2):int(len(data_y) / 8 * 3)].mean()
    f_pD_y3 = signal.welch(data_y, fs, nperseg=len(data_y))[1][int(len(data_y) / 8 * 3):int(len(data_y) / 2)].mean()

    f_pD_z0 = signal.welch(data_z, fs, nperseg=len(data_z))[1][0:int(len(data_z) / 8)].mean()
    f_pD_z1 = signal.welch(data_z, fs, nperseg=len(data_z))[1][int(len(data_z) / 8):int(len(data_z) / 8 * 2)].mean()
    f_pD_z2 = signal.welch(data_z, fs, nperseg=len(data_z))[1][int(len(data_z) / 8 * 2):int(len(data_z) / 8 * 3)].mean()
    f_pD_z3 = signal.welch(data_z, fs, nperseg=len(data_z))[1][int(len(data_z) / 8 * 3):int(len(data_z) / 2)].mean()

    f_pD_mag0 = signal.welch(data_mag, fs, nperseg=len(data_mag))[1][0:int(len(data_mag) / 8)].mean()
    f_pD_mag1 = signal.welch(data_mag, fs, nperseg=len(data_mag))[1][
                int(len(data_mag) / 8):int(len(data_mag) / 8 * 2)].mean()
    f_pD_mag2 = signal.welch(data_mag, fs, nperseg=len(data_mag))[1][
                int(len(data_mag) / 8 * 2):int(len(data_mag) / 8 * 3)].mean()
    f_pD_mag3 = signal.welch(data_mag, fs, nperseg=len(data_mag))[1][
                int(len(data_mag) / 8 * 3):int(len(data_mag) / 2)].mean()

    f_all = [f_rms_x, f_rms_y, f_rms_z, f_rms_mag, f_ptp_x, f_ptp_y, f_ptp_z, f_ptp_mag,
             f_pD_x0, f_pD_x1, f_pD_x2, f_pD_x3, f_pD_y0, f_pD_y1, f_pD_y2, f_pD_y3,
             f_pD_z0, f_pD_z1, f_pD_z2, f_pD_z3, f_pD_mag0, f_pD_mag1, f_pD_mag2, f_pD_mag3]


    return np.array(f_all)


def generate_all_features(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, fs):
    # Generate All Signal
    median_tAcc_X = median_filter(acc_x, f_size=3)
    median_tAcc_Y = median_filter(acc_y, f_size=3)
    median_tAcc_Z = median_filter(acc_z, f_size=3)

    butter_tAcc_X = butter_lowpass_filter(median_tAcc_X, cutoff=20, fs=fs, order=3)
    butter_tAcc_Y = butter_lowpass_filter(median_tAcc_Y, cutoff=20, fs=fs, order=3)
    butter_tAcc_Z = butter_lowpass_filter(median_tAcc_Z, cutoff=20, fs=fs, order=3)

    tBodyAcc_X = butter_highpass_filter(butter_tAcc_X, cutoff=0.3, fs=fs, order=3)
    tBodyAcc_Y = butter_highpass_filter(butter_tAcc_Y, cutoff=0.3, fs=fs, order=3)
    tBodyAcc_Z = butter_highpass_filter(butter_tAcc_Z, cutoff=0.3, fs=fs, order=3)

    tGravityAcc_X = butter_lowpass_filter(butter_tAcc_X, cutoff=0.3, fs=fs, order=3)
    tGravityAcc_Y = butter_lowpass_filter(butter_tAcc_Y, cutoff=0.3, fs=fs, order=3)
    tGravityAcc_Z = butter_lowpass_filter(butter_tAcc_Z, cutoff=0.3, fs=fs, order=3)

    tBodyAccJerk_X = calc_jerk(tBodyAcc_X)
    tBodyAccJerk_Y = calc_jerk(tBodyAcc_Y)
    tBodyAccJerk_Z = calc_jerk(tBodyAcc_Z)

    tBodyAccMag = np.vectorize(magnitude)(tBodyAcc_X, tBodyAcc_Y, tBodyAcc_Z)
    tGravityAccMag = np.vectorize(magnitude)(tGravityAcc_X, tGravityAcc_Y, tGravityAcc_Z)
    tBodyAccJerkMag = np.vectorize(magnitude)(tBodyAccJerk_X, tBodyAccJerk_Y, tBodyAccJerk_Z)

    fBodyAcc_X = calc_fft_signal(tBodyAcc_X)
    fBodyAcc_Y = calc_fft_signal(tBodyAcc_Y)
    fBodyAcc_Z = calc_fft_signal(tBodyAcc_Z)

    fBodyAccJerk_X = calc_fft_signal(tBodyAccJerk_X)
    fBodyAccJerk_Y = calc_fft_signal(tBodyAccJerk_Y)
    fBodyAccJerk_Z = calc_fft_signal(tBodyAccJerk_Z)

    fBodyAccMag = calc_fft_signal(tBodyAccMag)
    fBodyAccJerkMag = calc_fft_signal(tBodyAccJerkMag)

    median_tGyr_X = median_filter(gyr_x, 3)
    median_tGyr_Y = median_filter(gyr_y, 3)
    median_tGyr_Z = median_filter(gyr_z, 3)

    tBodyGyr_X = butter_lowpass_filter(median_tGyr_X, cutoff=20, fs=fs, order=3)
    tBodyGyr_Y = butter_lowpass_filter(median_tGyr_Y, cutoff=20, fs=fs, order=3)
    tBodyGyr_Z = butter_lowpass_filter(median_tGyr_Z, cutoff=20, fs=fs, order=3)

    tBodyGyrJerk_X = calc_jerk(tBodyGyr_X)
    tBodyGyrJerk_Y = calc_jerk(tBodyGyr_Y)
    tBodyGyrJerk_Z = calc_jerk(tBodyGyr_Z)

    tBodyGyrMag = np.vectorize(magnitude)(tBodyGyr_X, tBodyGyr_Y, tBodyGyr_Z)
    tBodyGyrJerkMag = np.vectorize(magnitude)(tBodyGyrJerk_X, tBodyGyrJerk_Y, tBodyGyrJerk_Z)

    fBodyGyr_X = calc_fft_signal(tBodyGyr_X)
    fBodyGyr_Y = calc_fft_signal(tBodyGyr_Y)
    fBodyGyr_Z = calc_fft_signal(tBodyGyr_Z)

    # fBodyGyrJerk_X = calc_fft_signal(tBodyGyrJerk_X)
    # fBodyGyrJerk_Y = calc_fft_signal(tBodyGyrJerk_Y)
    # fBodyGyrJerk_Z = calc_fft_signal(tBodyGyrJerk_Z)

    fBodyGyrMag = calc_fft_signal(tBodyGyrMag)
    fBodyGyrJerkMag = calc_fft_signal(tBodyGyrJerkMag)

    # Generate All Feature
    tBodyAcc_XYZ_features = time_xyz_feature_extraction(tBodyAcc_X, tBodyAcc_Y, tBodyAcc_Z)
    tGravityAcc_XYZ_features = time_xyz_feature_extraction(tGravityAcc_X, tGravityAcc_Y, tGravityAcc_Z)
    tBodyAccJerk_XYZ_features = time_xyz_feature_extraction(tBodyAccJerk_X, tBodyAccJerk_Y, tBodyAccJerk_Z)
    tBodyGyr_XYZ_features = time_xyz_feature_extraction(tBodyGyr_X, tBodyGyr_Y, tBodyGyr_Z)
    tBodyGyrJerk_XYZ_features = time_xyz_feature_extraction(tBodyGyrJerk_X, tBodyGyrJerk_Y, tBodyGyrJerk_Z)

    tBodyAccMag_features = time_mag_feature_extraction(tBodyAccMag)
    tGravityAccMag_features = time_mag_feature_extraction(tGravityAccMag)
    tBodyAccJerkMag_features = time_mag_feature_extraction(tBodyAccJerkMag)
    tBodyGyrMag_features = time_mag_feature_extraction(tBodyGyrMag)
    tBodyGyrJerkMag_features = time_mag_feature_extraction(tBodyGyrJerkMag)

    fBodyAcc_XYZ_features = frequency_xyz_feature_extraction(fBodyAcc_X, fBodyAcc_Y, fBodyAcc_Z, fs)
    fBodyAccJerk_XYZ_features = frequency_xyz_feature_extraction(fBodyAccJerk_X, fBodyAccJerk_Y, fBodyAccJerk_Z, fs)
    fBodyGyr_XYZ_features = frequency_xyz_feature_extraction(fBodyGyr_X, fBodyGyr_Y, fBodyGyr_Z, fs)
    fBodyGyrMag_features = frequency_mag_feature_extraction(fBodyGyrMag, fs)

    fBodyAccMag_features = frequency_mag_feature_extraction(fBodyAccMag, fs)
    fBodyAccJerkMag_features = frequency_mag_feature_extraction(fBodyAccJerkMag, fs)
    fBodyGyrJerkMag_features = frequency_mag_feature_extraction(fBodyGyrJerkMag, fs)

    angle_features = angle_feature_extraction(tBodyAcc_X, tBodyAcc_Y, tBodyAcc_Z, \
                                              tBodyAccJerk_X, tBodyAccJerk_Y, tBodyAccJerk_Z, \
                                              tBodyGyr_X, tBodyGyr_Y, tBodyGyr_Z, \
                                              tBodyGyrJerk_X, tBodyGyrJerk_Y, tBodyGyrJerk_Z, \
                                              tGravityAcc_X, tGravityAcc_Y, tGravityAcc_Z)

    tBodyAcc_additional_features = additional_feature_extraction(tBodyAcc_X, tBodyAcc_Y, tBodyAcc_Z, tBodyAccMag, fs)
    tBodyAccJerk_additional_features = additional_feature_extraction(tBodyAccJerk_X, tBodyAccJerk_Y, tBodyAccJerk_Z,
                                                                     tBodyAccJerkMag, fs)
    tBodyGyro_additional_features = additional_feature_extraction(tBodyGyr_X, tBodyGyr_Y, tBodyGyr_Z, tBodyGyrMag, fs)

    all_features = np.hstack([
        tBodyAcc_XYZ_features,
        tGravityAcc_XYZ_features,
        tBodyAccJerk_XYZ_features,
        tBodyGyr_XYZ_features,
        tBodyGyrJerk_XYZ_features,
        tBodyAccMag_features,
        tGravityAccMag_features,
        tBodyAccJerkMag_features,
        tBodyGyrMag_features,
        tBodyGyrJerkMag_features,
        fBodyAcc_XYZ_features,
        fBodyAccJerk_XYZ_features,
        fBodyGyr_XYZ_features,
        fBodyAccMag_features,
        fBodyAccJerkMag_features,
        fBodyGyrMag_features,
        fBodyGyrJerkMag_features,
        angle_features,
        tBodyAcc_additional_features,
        tBodyAccJerk_additional_features,
        tBodyGyro_additional_features
    ]).T

    return all_features


def generate_all_features_561(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, fs):
    # Generate All Signal
    median_tAcc_X = median_filter(acc_x, f_size=3)
    median_tAcc_Y = median_filter(acc_y, f_size=3)
    median_tAcc_Z = median_filter(acc_z, f_size=3)

    butter_tAcc_X = butter_lowpass_filter(median_tAcc_X, cutoff=20, fs=fs, order=3)
    butter_tAcc_Y = butter_lowpass_filter(median_tAcc_Y, cutoff=20, fs=fs, order=3)
    butter_tAcc_Z = butter_lowpass_filter(median_tAcc_Z, cutoff=20, fs=fs, order=3)

    tBodyAcc_X = butter_highpass_filter(butter_tAcc_X, cutoff=0.3, fs=fs, order=3)
    tBodyAcc_Y = butter_highpass_filter(butter_tAcc_Y, cutoff=0.3, fs=fs, order=3)
    tBodyAcc_Z = butter_highpass_filter(butter_tAcc_Z, cutoff=0.3, fs=fs, order=3)

    tGravityAcc_X = butter_lowpass_filter(butter_tAcc_X, cutoff=0.3, fs=fs, order=3)
    tGravityAcc_Y = butter_lowpass_filter(butter_tAcc_Y, cutoff=0.3, fs=fs, order=3)
    tGravityAcc_Z = butter_lowpass_filter(butter_tAcc_Z, cutoff=0.3, fs=fs, order=3)

    tBodyAccJerk_X = calc_jerk(tBodyAcc_X)
    tBodyAccJerk_Y = calc_jerk(tBodyAcc_Y)
    tBodyAccJerk_Z = calc_jerk(tBodyAcc_Z)

    tBodyAccMag = np.vectorize(magnitude)(tBodyAcc_X, tBodyAcc_Y, tBodyAcc_Z)
    tGravityAccMag = np.vectorize(magnitude)(tGravityAcc_X, tGravityAcc_Y, tGravityAcc_Z)
    tBodyAccJerkMag = np.vectorize(magnitude)(tBodyAccJerk_X, tBodyAccJerk_Y, tBodyAccJerk_Z)

    fBodyAcc_X = calc_fft_signal(tBodyAcc_X)
    fBodyAcc_Y = calc_fft_signal(tBodyAcc_Y)
    fBodyAcc_Z = calc_fft_signal(tBodyAcc_Z)

    fBodyAccJerk_X = calc_fft_signal(tBodyAccJerk_X)
    fBodyAccJerk_Y = calc_fft_signal(tBodyAccJerk_Y)
    fBodyAccJerk_Z = calc_fft_signal(tBodyAccJerk_Z)

    fBodyAccMag = calc_fft_signal(tBodyAccMag)
    fBodyAccJerkMag = calc_fft_signal(tBodyAccJerkMag)

    median_tGyr_X = median_filter(gyr_x, 3)
    median_tGyr_Y = median_filter(gyr_y, 3)
    median_tGyr_Z = median_filter(gyr_z, 3)

    tBodyGyr_X = butter_lowpass_filter(median_tGyr_X, cutoff=20, fs=fs, order=3)
    tBodyGyr_Y = butter_lowpass_filter(median_tGyr_Y, cutoff=20, fs=fs, order=3)
    tBodyGyr_Z = butter_lowpass_filter(median_tGyr_Z, cutoff=20, fs=fs, order=3)

    tBodyGyrJerk_X = calc_jerk(tBodyGyr_X)
    tBodyGyrJerk_Y = calc_jerk(tBodyGyr_Y)
    tBodyGyrJerk_Z = calc_jerk(tBodyGyr_Z)

    tBodyGyrMag = np.vectorize(magnitude)(tBodyGyr_X, tBodyGyr_Y, tBodyGyr_Z)
    tBodyGyrJerkMag = np.vectorize(magnitude)(tBodyGyrJerk_X, tBodyGyrJerk_Y, tBodyGyrJerk_Z)

    fBodyGyr_X = calc_fft_signal(tBodyGyr_X)
    fBodyGyr_Y = calc_fft_signal(tBodyGyr_Y)
    fBodyGyr_Z = calc_fft_signal(tBodyGyr_Z)

    # fBodyGyrJerk_X = calc_fft_signal(tBodyGyrJerk_X, fs, window)
    # fBodyGyrJerk_Y = calc_fft_signal(tBodyGyrJerk_Y, fs, window)
    # fBodyGyrJerk_Z = calc_fft_signal(tBodyGyrJerk_Z, fs, window)

    fBodyGyrMag = calc_fft_signal(tBodyGyrMag)
    fBodyGyrJerkMag = calc_fft_signal(tBodyGyrJerkMag)

    # Generate All Feature
    tBodyAcc_XYZ_features = time_xyz_feature_extraction(tBodyAcc_X, tBodyAcc_Y, tBodyAcc_Z)
    tGravityAcc_XYZ_features = time_xyz_feature_extraction(tGravityAcc_X, tGravityAcc_Y, tGravityAcc_Z)
    tBodyAccJerk_XYZ_features = time_xyz_feature_extraction(tBodyAccJerk_X, tBodyAccJerk_Y, tBodyAccJerk_Z)
    tBodyGyr_XYZ_features = time_xyz_feature_extraction(tBodyGyr_X, tBodyGyr_Y, tBodyGyr_Z)
    tBodyGyrJerk_XYZ_features = time_xyz_feature_extraction(tBodyGyrJerk_X, tBodyGyrJerk_Y, tBodyGyrJerk_Z)

    tBodyAccMag_features = time_mag_feature_extraction(tBodyAccMag)
    tGravityAccMag_features = time_mag_feature_extraction(tGravityAccMag)
    tBodyAccJerkMag_features = time_mag_feature_extraction(tBodyAccJerkMag)
    tBodyGyrMag_features = time_mag_feature_extraction(tBodyGyrMag)
    tBodyGyrJerkMag_features = time_mag_feature_extraction(tBodyGyrJerkMag)

    fBodyAcc_XYZ_features = frequency_xyz_feature_extraction(fBodyAcc_X, fBodyAcc_Y, fBodyAcc_Z, fs)
    fBodyAccJerk_XYZ_features = frequency_xyz_feature_extraction(fBodyAccJerk_X, fBodyAccJerk_Y, fBodyAccJerk_Z, fs)
    fBodyGyr_XYZ_features = frequency_xyz_feature_extraction(fBodyGyr_X, fBodyGyr_Y, fBodyGyr_Z, fs)
    fBodyGyrMag_features = frequency_mag_feature_extraction(fBodyGyrMag, fs)

    fBodyAccMag_features = frequency_mag_feature_extraction(fBodyAccMag, fs)
    fBodyAccJerkMag_features = frequency_mag_feature_extraction(fBodyAccJerkMag, fs)
    fBodyGyrJerkMag_features = frequency_mag_feature_extraction(fBodyGyrJerkMag, fs)

    angle_features = angle_feature_extraction(tBodyAcc_X, tBodyAcc_Y, tBodyAcc_Z, \
                                              tBodyAccJerk_X, tBodyAccJerk_Y, tBodyAccJerk_Z, \
                                              tBodyGyr_X, tBodyGyr_Y, tBodyGyr_Z, \
                                              tBodyGyrJerk_X, tBodyGyrJerk_Y, tBodyGyrJerk_Z, \
                                              tGravityAcc_X, tGravityAcc_Y, tGravityAcc_Z)


    all_features = np.hstack([
        tBodyAcc_XYZ_features,
        tGravityAcc_XYZ_features,
        tBodyAccJerk_XYZ_features,
        tBodyGyr_XYZ_features,
        tBodyGyrJerk_XYZ_features,
        tBodyAccMag_features,
        tGravityAccMag_features,
        tBodyAccJerkMag_features,
        tBodyGyrMag_features,
        tBodyGyrJerkMag_features,
        fBodyAcc_XYZ_features,
        fBodyAccJerk_XYZ_features,
        fBodyGyr_XYZ_features,
        fBodyAccMag_features,
        fBodyAccJerkMag_features,
        fBodyGyrMag_features,
        fBodyGyrJerkMag_features,
        angle_features
    ]).T

    return all_features

def compute_all_features(tAcc_XYZ, tGyro_XYZ, window, slide, fs):
    sliding_tAcc_X = sliding_window(tAcc_XYZ[:, 0], window, slide)
    sliding_tAcc_Y = sliding_window(tAcc_XYZ[:, 1], window, slide)
    sliding_tAcc_Z = sliding_window(tAcc_XYZ[:, 2], window, slide)

    sliding_tGyr_X = sliding_window(tGyro_XYZ[:, 0], window, slide)
    sliding_tGyr_Y = sliding_window(tGyro_XYZ[:, 1], window, slide)
    sliding_tGyr_Z = sliding_window(tGyro_XYZ[:, 2], window, slide)

    N = len(sliding_tAcc_X)

    all_features = np.asarray([ \
        generate_all_features_561(sliding_tAcc_X[i], sliding_tAcc_Y[i], sliding_tAcc_Z[i], sliding_tGyr_X[i],
                              sliding_tGyr_Y[i], sliding_tGyr_Z[i], fs) \
        for i in range(N)])

    return all_features
