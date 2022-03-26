import os
import numpy as np
import pandas as pd
import glob
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

PEAKS = [965.7786, 978.4503, 1013.975, 1047.0054, 1067.3565, 1128.71, 1148.8109, 1166.871, 1211.2326, 1234.3393,
         1240.2827, 1243.9321, 1248.7663, 1270.2281, 1280.2739, 1295.6659, 1300.8264, 1321.399, 1327.264, 1331.321,
         1336.7111, 1350.4191, 1362.2659, 1371.8577, 1382.5715, 1409.364, 1529.582]


def main():
    filename_calibration = input('キャリブレーションに使うascファイルのパス: ')

    df_calibration = pd.read_csv(filename_calibration, index_col=0, delimiter='\t', header=None).reset_index(drop=True)
    x, y = df_calibration.T.values

    peaks, _ = find_peaks(y, height=y.mean())

    # plt.plot(x, y, color='c')
    # plt.plot(x[peaks], y[peaks], 'x', color='blue')
    # plt.axhline(y=y.mean())
    # plt.title('Peaks')

    x_detected = np.array([])
    peaks_detected = np.array([])

    search_width = 3

    for peak in peaks:
        i = 0
        while i < len(PEAKS):
            if abs(PEAKS[i] - x[peak]) <= search_width and abs(PEAKS[i] - x[peak]) < min([abs(PEAKS[i-1] - x[peak]), abs(PEAKS[i+1] - x[peak])]):
                x_detected = np.append(x_detected, peak)
                peaks_detected = np.append(peaks_detected, PEAKS[i])
                break
            i += 1

    print('\nキャリブレーションに使用するピーク: ')
    print('番号 実測値 → 校正値')
    for i, peak in enumerate(x_detected):
        print(f'{i+1} {round(x[int(peak)], 1)} → {round(peaks_detected[i], 1)}')
    print()

    # plt.show()

    cubic = PolynomialFeatures(degree=3)  # 特徴量の冪乗を特徴量に追加するための関数

    x_detected_reshaped = x_detected.reshape([x_detected.size, -1])  # 特徴量ベクトルを転置
    x_detected_cubic = cubic.fit_transform(x_detected_reshaped)

    model = LinearRegression()  # 線形回帰
    model.fit(x_detected_cubic, peaks_detected)

    print('-'*50)
    print(f'回帰モデルの係数: {model.coef_}')
    peaks_pred = model.predict(x_detected_cubic)
    print(f'回帰モデルのスコア: {r2_score(peaks_detected, peaks_pred)}')
    print('-'*50)

    x = np.arange(0, 512)
    x_reshaped = x.reshape([x.size, -1])
    x_cubic = cubic.fit_transform(x_reshaped)
    x_calibrated = model.predict(x_cubic)

    folder = input('キャリブレーションしたいascファイルの入ったフォルダのパス: ')

    if not os.path.exists(folder + '_calibrated'):
        os.mkdir(folder + '_calibrated')

    filenames = glob.glob(os.path.join(folder, '*.asc'))
    for filename in filenames:
        df_tmp = pd.read_csv(filename, index_col=0, delimiter='\t', header=None)
        df_tmp.columns = ['x', 'y']
        df_tmp.x = x_calibrated
        filename_new = folder + '_calibrated' + os.sep + filename.split(os.sep)[-1][:-4] + '_calibrated.asc'
        df_tmp.to_csv(filename_new, sep='\t', header=None)

    print('\nキャリブレーションをしたデータが入ったフォルダが作成されました')


if __name__ == '__main__':
    main()
