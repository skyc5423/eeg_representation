from network.network import AutoEncoder
import mne
import os
import pyedflib
import datetime
import numpy as np
import math
import json
from scipy.stats import gamma


class Util(object):
    YB_CH_NAME_LIST = ['Fp1', 'F7', 'T3', 'T5', 'T6', 'T4', 'F8', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'Pz', 'Fz', 'Cz', 'O1', 'P3', 'C3', 'F3']
    YB_CH_SET_LIST = {'single': {'frontal': ['Fp1', 'Fp2', 'F3', 'Fz', 'F4'],
                                 'left_temporal': ['F7', 'T3', 'T5'],
                                 'right_temporal': ['F8', 'T4', 'T6'],
                                 'parietal': ['C3', 'C4', 'P3', 'P4', 'Pz'],
                                 'occipital': ['O1', 'O2'],
                                 'central': ['Cz'],
                                 },
                      'connectivity': {'interhemispheric': [('Fp1', 'Fp2'), ('F3', 'F4'), ('F7', 'F8'), ('C3', 'C4'),
                                                            ('T3', 'T4'), ('T5', 'T6'), ('P3', 'P4'), ('O1', 'O2')],
                                       'intrahemispheric': [('F3', 'T3'), ('F3', 'C3'), ('F3', 'O1'), ('T3', 'C3'),
                                                            ('T3', 'O1'), ('C3', 'O1'), ('F4', 'T4'), ('F4', 'C4'),
                                                            ('F4', 'O2'), ('T4', 'C4'), ('T4', 'O2'), ('C4', 'O2')]}

                      }

    @staticmethod
    def parse_network(pretrain_dir: str):
        f = open(('./%s/network_info.txt' % pretrain_dir), 'r')
        info_lines = f.readlines()
        kernel_size_list = []
        for i in range(int(len(info_lines) / 2)):
            if 'dense/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'dense_1/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'dense_2/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'dense_3/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'dense_4/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'discriminator_1/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'discriminator_2/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'discriminator_3/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))

        return kernel_size_list

    @staticmethod
    def write_edf(mne_raw, fname, picks=None, tmin=0, tmax=None, overwrite=False):
        if not issubclass(type(mne_raw), mne.io.BaseRaw):
            raise TypeError('Must be mne.io.Raw type')
        if not overwrite and os.path.exists(fname):
            return
            raise OSError('File already exists. No overwrite.')
        # static settings
        file_type = pyedflib.FILETYPE_EDFPLUS
        sfreq = mne_raw.info['sfreq']
        first_sample = int(sfreq * tmin)
        last_sample = int(sfreq * tmax) if tmax is not None else None

        # convert data
        channels = mne_raw.get_data(picks,
                                    start=first_sample,
                                    stop=last_sample)

        # set conversion parameters
        dmin, dmax = [-32768, 32767]
        pmin, pmax = [channels.min(), channels.max()]
        n_channels = len(channels)

        # create channel from this
        try:
            f = pyedflib.EdfWriter(fname,
                                   n_channels=n_channels,
                                   file_type=file_type)

            channel_info = []
            data_list = []

            for i in range(n_channels):
                ch_dict = {'label': mne_raw.ch_names[i],
                           'dimension': 'uV',
                           'sample_rate': sfreq,
                           'physical_min': pmin,
                           'physical_max': pmax,
                           'digital_min': dmin,
                           'digital_max': dmax,
                           'transducer': '',
                           'prefilter': ''}

                channel_info.append(ch_dict)
                data_list.append(channels[i])

            f.setTechnician('mne-gist-save-edf-skjerns')
            f.setSignalHeaders(channel_info)
            f.writeSamples(data_list)
        except Exception as e:
            print(e)
            return False
        f.close()
        return True

    @staticmethod
    def detect_bad_channel(closed_data):
        bad_ch_list = []

        for ch in range(closed_data.shape[0]):
            if np.min(closed_data[ch]) > -10 and np.max(closed_data[ch]) < 10:
                bad_ch_list.append(ch)

        return bad_ch_list

    @staticmethod
    def get_neighbor_ch_list(hf, child_db_ch_index_list):

        # chan_loc_f = open('/home/ybrain-analysis/문서/dataset/child_data/GSN_HydroCel_129.sfp.txt', 'r')
        # chan_loc_lines = chan_loc_f.readlines()

        neighbor_ch_list = []
        for idx in child_db_ch_index_list:
            crt_coord_x = np.array(hf[hf['EEG']['chanlocs']['X'][idx][0]])[0][0]
            crt_coord_y = np.array(hf[hf['EEG']['chanlocs']['Y'][idx][0]])[0][0]
            crt_coord_z = np.array(hf[hf['EEG']['chanlocs']['Z'][idx][0]])[0][0]
            # crt_coord = chan_loc_lines[idx + 3].split('\n')[0]
            # crt_coord = crt_coord.split('\t')[1:4]
            # crt_coord_x = float(crt_coord[0])
            # crt_coord_y = float(crt_coord[1])
            # crt_coord_z = float(crt_coord[2])
            dist_list = []
            for i in range(len(hf['EEG']['chanlocs']['X'])):
                cmp_coord_x = np.array(hf[hf['EEG']['chanlocs']['X'][i][0]])[0][0]
                cmp_coord_y = np.array(hf[hf['EEG']['chanlocs']['Y'][i][0]])[0][0]
                cmp_coord_z = np.array(hf[hf['EEG']['chanlocs']['Z'][i][0]])[0][0]
                # cmp_coord = chan_loc_lines[i].split('\n')[0]
                # cmp_coord = cmp_coord.split('\t')[1:4]
                # cmp_coord_x = float(cmp_coord[0])
                # cmp_coord_y = float(cmp_coord[1])
                # cmp_coord_z = float(cmp_coord[2])
                dist = np.sqrt(np.square(crt_coord_x - cmp_coord_x) + np.square(crt_coord_y - cmp_coord_y) + np.square(crt_coord_z - cmp_coord_z))
                dist_list.append(dist)
            dist_arr = np.array(dist_list)
            sort_idx = np.argsort(dist_arr)
            neighbor_ch_list.append(sort_idx[0:4])
        return neighbor_ch_list

    @staticmethod
    def write_edf(mne_raw, fname, picks=None, tmin=0, tmax=None, overwrite=False):
        if not issubclass(type(mne_raw), mne.io.BaseRaw):
            raise TypeError('Must be mne.io.Raw type')
        if not overwrite and os.path.exists(fname):
            return
            raise OSError('File already exists. No overwrite.')
        # static settings
        file_type = pyedflib.FILETYPE_EDFPLUS
        sfreq = mne_raw.info['sfreq']
        # date = datetime.now().strftime('%d %b %Y %H:%M:%S')
        first_sample = int(sfreq * tmin)
        last_sample = int(sfreq * tmax) if tmax is not None else None

        # convert data
        channels = mne_raw.get_data(picks,
                                    start=first_sample,
                                    stop=last_sample)

        # convert to microvolts to scale up precision
        # channels *= 1e6

        # set conversion parameters
        dmin, dmax = [-32768, 32767]
        pmin, pmax = [channels.min(), channels.max()]
        n_channels = len(channels)

        # create channel from this
        try:
            f = pyedflib.EdfWriter(fname,
                                   n_channels=n_channels,
                                   file_type=file_type)

            channel_info = []
            data_list = []

            for i in range(n_channels):
                ch_dict = {'label': mne_raw.ch_names[i],
                           'dimension': 'uV',
                           'sample_rate': sfreq,
                           'physical_min': pmin,
                           'physical_max': pmax,
                           'digital_min': dmin,
                           'digital_max': dmax,
                           'transducer': '',
                           'prefilter': ''}

                channel_info.append(ch_dict)
                data_list.append(channels[i])

            f.setTechnician('mne-gist-save-edf-skjerns')
            f.setSignalHeaders(channel_info)
            # f.setStartdatetime(date)
            f.writeSamples(data_list)
        except Exception as e:
            print(e)
            return False
        finally:
            f.close()
        return True

    def get_power_feature_from_json(self, json_file_path, ch_gathering, asymmetry, scale=False):
        our_ch_list = self.YB_CH_NAME_LIST
        try:
            f = open(json_file_path)
        except:
            return None, None
        jf = json.load(f)
        feature_val_list = []
        feature_name_list = []
        for key_feature in ['abs_power', 'rel_power', 'rat_power']:
            if jf[key_feature] is None:
                continue
            if key_feature == 'rat_power':
                band_set = ['DAR', 'TAR', 'TBR']
            else:
                band_set = ['Delta', 'Theta', 'Alpha', 'Beta', 'High Beta']
            for key_band in band_set:
                if ch_gathering:
                    feature_val = jf[key_feature][key_band]
                    if np.mean(feature_val) != np.mean(feature_val):
                        return None, None
                    for region in self.YB_CH_SET_LIST['single']:
                        total_power = 0
                        for ch_name in self.YB_CH_SET_LIST['single'][region]:
                            total_power += feature_val[self.YB_CH_NAME_LIST.index(ch_name)]
                        if key_feature == 'abs_power':
                            if scale:
                                feature_val_list.append(np.log(1E12 * total_power) / len(self.YB_CH_SET_LIST['single'][region]))
                            else:
                                feature_val_list.append(np.log(total_power) / len(self.YB_CH_SET_LIST['single'][region]))
                        elif key_feature == 'rel_power':
                            feature_val_list.append(np.log(total_power * 100) / len(self.YB_CH_SET_LIST['single'][region]))
                        else:
                            feature_val_list.append(np.log(total_power) / len(self.YB_CH_SET_LIST['single'][region]))
                        feature_name_list.append('%s_%s_%s' % (key_feature, key_band, region))
                else:
                    feature_val = jf[key_feature][key_band]
                    if np.mean(feature_val) != np.mean(feature_val):
                        return None, None
                    if key_feature == 'abs_power':
                        if scale:
                            feature_val_list += list(1E12 * np.array(feature_val))
                        else:
                            feature_val_list += feature_val
                    elif key_feature == 'rel_power':
                        feature_val_list += list(100 * np.array(feature_val))
                    else:
                        feature_val_list += list(np.array(feature_val))
                    for ch in range(19):
                        feature_name_list.append('%s_%s_%s' % (key_feature, key_band, our_ch_list[ch]))
        if asymmetry:
            for key_feature in ['abs_power']:
                if jf[key_feature] is None:
                    continue
                for key_band in ['Delta', 'Theta', 'Alpha', 'Beta', 'High Beta']:
                    feature_val = jf[key_feature][key_band]
                    if np.mean(feature_val) != np.mean(feature_val):
                        return None, None
                    for region in self.YB_CH_SET_LIST['connectivity']:
                        for (ch1, ch2) in self.YB_CH_SET_LIST['connectivity'][region]:
                            if scale:
                                feature_val_list.append((np.log(1E12 * feature_val[self.YB_CH_NAME_LIST.index(ch1)]) - np.log(1E12 * feature_val[self.YB_CH_NAME_LIST.index(ch2)])))
                            else:
                                feature_val_list.append((np.log(feature_val[self.YB_CH_NAME_LIST.index(ch1)]) - np.log(feature_val[self.YB_CH_NAME_LIST.index(ch2)])))
                            feature_name_list.append('%s_%s_%s-%s' % ('asymmetry', key_band, ch1, ch2))
        return np.array(feature_val_list), feature_name_list

    def get_coh_feature_from_json(self, json_file_path):
        try:
            f = open(json_file_path)
        except:
            return None, None
        jf = json.load(f)
        feature_val_list = []
        feature_name_list = []
        for key_feature in jf.keys():
            if jf[key_feature] is None:
                continue
            for key_band in jf[key_feature].keys():
                feature_val = np.array(jf[key_feature][key_band])
                if np.mean(feature_val) != np.mean(feature_val):
                    return None, None
                for ch_set in self.YB_CH_SET_LIST['connectivity']:
                    for (ch_1, ch_2) in self.YB_CH_SET_LIST['connectivity'][ch_set]:
                        feature_val_list.append(feature_val[self.YB_CH_NAME_LIST.index(ch_1), self.YB_CH_NAME_LIST.index(ch_2)])
                        feature_name_list.append('%s_%s_%s_%s' % (key_feature, key_band, ch_1, ch_2))
        return np.array(feature_val_list), feature_name_list

    @staticmethod
    def get_gamma_dist(input_list):

        N = input_list.__len__()
        tmp_a = 0
        tmp_b = 0
        tmp_c = 0
        tmp_d = 0
        for i in range(N):
            tmp_a += input_list[i]
            tmp_b += (input_list[i] * math.log(input_list[i] + 1E-7))
            tmp_c += math.log(input_list[i] + 1E-7)
            tmp_d += input_list[i]
        tmp_a *= N
        tmp_b *= N
        tmp_e = tmp_c * tmp_d
        k = tmp_a / (tmp_b - tmp_e + 1E-7)
        theta = (tmp_b - tmp_e) / (N * N + 1E-7)

        return gamma(a=k, scale=theta), k, theta
