from .feature_extraction import *
import numpy as np

# Maybe another function that just generates the gravity for the additional anhgle feature extraction 
#   of the gyro?

def acc_angle_feature_extraction(bodyacc_x, bodyacc_y, bodyacc_z, \
                             bodyaccJerk_x, bodyaccJerk_y, bodyaccJerk_z, \
                             gravity_x, gravity_y, gravity_z):
    tBodyAcc_Gravity = calc_angle([bodyacc_x.mean(), bodyacc_y.mean(), bodyacc_z.mean()], \
                                  [gravity_x.mean(), gravity_y.mean(), gravity_z.mean()])
    tBodyAccJerk_Gravity = calc_angle([bodyaccJerk_x.mean(), bodyaccJerk_y.mean(), bodyaccJerk_z.mean()], \
                                      [gravity_x.mean(), gravity_y.mean(), gravity_z.mean()])
    tBodyAcc_X = calc_angle([1, 0, 0], [gravity_x.mean(), gravity_y.mean(), gravity_z.mean()])
    tBodyAcc_Y = calc_angle([0, 1, 0], [gravity_x.mean(), gravity_y.mean(), gravity_z.mean()])
    tBodyAcc_Z = calc_angle([0, 0, 1], [gravity_x.mean(), gravity_y.mean(), gravity_z.mean()])
    
    f_acc = [tBodyAcc_Gravity, tBodyAccJerk_Gravity, tBodyAcc_X, tBodyAcc_Y, tBodyAcc_Z]

    return np.array(f_acc)

def generate_all_Gyr_features(gyr_x, gyr_y, gyr_z, fs):
    median_tGyr_X = median_filter(gyr_x, 3)
    median_tGyr_Y = median_filter(gyr_y, 3)
    median_tGyr_Z = median_filter(gyr_z, 3)

    tBodyGyr_X = butter_lowpass_filter(median_tGyr_X, cutoff=20, fs=fs, order=3)
    tBodyGyr_Y = butter_lowpass_filter(median_tGyr_Y, cutoff=20, fs=fs, order=3)
    tBodyGyr_Z = butter_lowpass_filter(median_tGyr_Z, cutoff=20, fs=fs, order=3)

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

    fBodyGyrMag = calc_fft_signal(tBodyGyrMag)
    fBodyGyrJerkMag = calc_fft_signal(tBodyGyrJerkMag)

    tBodyGyr_XYZ_features = time_xyz_feature_extraction(tBodyGyr_X, tBodyGyr_Y, tBodyGyr_Z)
    tBodyGyrJerk_XYZ_features = time_xyz_feature_extraction(tBodyGyrJerk_X, tBodyGyrJerk_Y, tBodyGyrJerk_Z)

    tBodyGyrMag_features = time_mag_feature_extraction(tBodyGyrMag)
    tBodyGyrJerkMag_features = time_mag_feature_extraction(tBodyGyrJerkMag)

    fBodyGyr_XYZ_features = frequency_xyz_feature_extraction(fBodyGyr_X, fBodyGyr_Y, fBodyGyr_Z, fs)
    fBodyGyrMag_features = frequency_mag_feature_extraction(fBodyGyrMag, fs)

    fBodyGyrJerkMag_features = frequency_mag_feature_extraction(fBodyGyrJerkMag, fs)

    tBodyGyro_additional_features = additional_feature_extraction(tBodyGyr_X, tBodyGyr_Y, tBodyGyr_Z, tBodyGyrMag, fs)

    all_gyr_features = np.hstack([
        tBodyGyr_XYZ_features,
        tBodyGyrJerk_XYZ_features,
        tBodyGyrMag_features,
        tBodyGyrJerkMag_features,
        fBodyGyr_XYZ_features,
        fBodyGyrMag_features,
        fBodyGyrJerkMag_features,
        # tBodyGyro_additional_features
    ]).T

    return all_gyr_features

def generate_all_Acc_features(acc_x, acc_y, acc_z, fs):
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

    tBodyAcc_XYZ_features = time_xyz_feature_extraction(tBodyAcc_X, tBodyAcc_Y, tBodyAcc_Z)
    tGravityAcc_XYZ_features = time_xyz_feature_extraction(tGravityAcc_X, tGravityAcc_Y, tGravityAcc_Z)
    tBodyAccJerk_XYZ_features = time_xyz_feature_extraction(tBodyAccJerk_X, tBodyAccJerk_Y, tBodyAccJerk_Z)

    tBodyAccMag_features = time_mag_feature_extraction(tBodyAccMag)
    tGravityAccMag_features = time_mag_feature_extraction(tGravityAccMag)
    tBodyAccJerkMag_features = time_mag_feature_extraction(tBodyAccJerkMag)

    fBodyAcc_XYZ_features = frequency_xyz_feature_extraction(fBodyAcc_X, fBodyAcc_Y, fBodyAcc_Z, fs)
    fBodyAccJerk_XYZ_features = frequency_xyz_feature_extraction(fBodyAccJerk_X, fBodyAccJerk_Y, fBodyAccJerk_Z, fs)

    fBodyAccMag_features = frequency_mag_feature_extraction(fBodyAccMag, fs)
    fBodyAccJerkMag_features = frequency_mag_feature_extraction(fBodyAccJerkMag, fs)

    acc_angle_features = acc_angle_feature_extraction(tBodyAcc_X, tBodyAcc_Y, tBodyAcc_Z, \
                                              tBodyAccJerk_X, tBodyAccJerk_Y, tBodyAccJerk_Z, \
                                              tGravityAcc_X, tGravityAcc_Y, tGravityAcc_Z)

    tBodyAcc_additional_features = additional_feature_extraction(tBodyAcc_X, tBodyAcc_Y, tBodyAcc_Z, tBodyAccMag, fs)
    tBodyAccJerk_additional_features = additional_feature_extraction(tBodyAccJerk_X, tBodyAccJerk_Y, tBodyAccJerk_Z, tBodyAccJerkMag, fs)

    all_acc_features = np.hstack([
        tBodyAcc_XYZ_features,
        tGravityAcc_XYZ_features,
        tBodyAccJerk_XYZ_features,
        tBodyAccMag_features,
        tGravityAccMag_features,
        tBodyAccJerkMag_features,
        fBodyAcc_XYZ_features,
        fBodyAccJerk_XYZ_features,
        fBodyAccMag_features,
        fBodyAccJerkMag_features,
        acc_angle_features,
        # tBodyAcc_additional_features,
        # tBodyAccJerk_additional_features
    ]).T

    return all_acc_features

def compute_all_Gyr_features(tGyr_XYZ, window, slide, fs):
    sliding_tGyr_X = sliding_window(tGyr_XYZ[:, 0], window, slide)
    sliding_tGyr_Y = sliding_window(tGyr_XYZ[:, 1], window, slide)
    sliding_tGyr_Z = sliding_window(tGyr_XYZ[:, 2], window, slide)
    
    N = len(sliding_tGyr_X)
    all_Gyr_features = np.asarray([\
    generate_all_Gyr_features(
            sliding_tGyr_X[i], 
            sliding_tGyr_Y[i],
            sliding_tGyr_Z[i], 
            fs
        ) for i in range(N)])
    
    return all_Gyr_features

def compute_all_Acc_features(tAcc_XYZ, window, slide, fs):
    sliding_tAcc_X = sliding_window(tAcc_XYZ[:, 0], window, slide)
    sliding_tAcc_Y = sliding_window(tAcc_XYZ[:, 1], window, slide)
    sliding_tAcc_Z = sliding_window(tAcc_XYZ[:, 2], window, slide)
    
    N = len(sliding_tAcc_X)
    all_Acc_features = np.asarray([\
    generate_all_Acc_features(
            sliding_tAcc_X[i], 
            sliding_tAcc_Y[i],
            sliding_tAcc_Z[i], 
            fs
        ) for i in range(N)])
    
    return all_Acc_features

