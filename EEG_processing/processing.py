import mne
import matplotlib.pyplot as plt
import numpy as np
def EEG_processing(path,save_path):
    # 读取FIF文件
    file_path = path
    raw = mne.io.read_raw_fif(file_path, preload=True,)
    raw.pick(picks='all', exclude=['ECG'])
    raw.pick_types(eeg=True, stim=True, eog=True)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    # 绘制原始脑电图
    raw.plot()
    plt.savefig(save_path + 'raw_eeg_plot.jpg')
    # plt.show()
    # 绘制拓扑图
    layout = mne.channels.find_layout(raw.info, ch_type='eeg')
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(raw.get_data()[:, 0], layout.pos[:, :2], axes=ax, show=False)
    plt.savefig(save_path + 'topography_plot.png')
    # plt.show()

if __name__=="__main__":
    EEG_processing('Anger_EGG_1.fif',"")