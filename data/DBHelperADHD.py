import numpy as np
import os
import pandas as pd
import json
import math
from scipy.stats import gamma, norm, beta
from util import Util
from pathlib import Path

from config.cfg import cfg


class DBHelper(object):
    def __init__(self, data_path):
        self.hbn_data_path = Path(data_path) / Path(cfg.DATA.HBN_PATH)
        self.adhd_data_path = Path(data_path) / Path(cfg.DATA.SOONCHEONHYANG_PATH)
        self.mipdb_data_path = Path(data_path) / Path(cfg.DATA.MIPDB_PATH)

        self.total_data = None
        self.norm_total_data = None
        self.hbn_data = None
        self.adhd_data = None
        self.mipdb_data = None

        self.hbn_label = None
        self.adhd_label = None
        self.mipdb_label = None

        self.hbn_age = None
        self.adhd_age = None
        self.mipdb_age = None
        self.mdd_age = None
        self.ccss_age = None
        self.total_age = None

        self.adhd_start_idx = -1
        self.mipdb_start_idx = -1
        self.mdd_start_idx = -1
        self.ccss_start_idx = -1

        self.feature_name_list = None
        self.YB_CH_NAME_LIST = ['Fp1', 'F7', 'T3', 'T5', 'T6', 'T4', 'F8', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'Pz', 'Fz', 'Cz', 'O1', 'P3', 'C3', 'F3']

    def age_normalize_feature_neuroguide(self, delete_cz=True):
        self.age_norm_set = [(0, 9),
                             (9, 12),
                             (12, 15),
                             (15, 20),
                             (20, 30),
                             (30, 80)
                             ]
        age_range_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 45, 999]

        norm_arr = np.load('./data/ref_file/norm.npy')

        z_score = np.zeros_like(self.total_data)

        for age_idx in range(25):
            age_select = np.where((self.total_age >= age_range_list[age_idx]) & (self.total_age < age_range_list[age_idx + 1]))[0]
            for feature_idx in range(13):
                for ch_idx in range(19):
                    z_score[age_select, feature_idx * 19 + ch_idx] = norm_arr[feature_idx, ch_idx, age_idx, 0] * np.log(
                        self.total_data[age_select, feature_idx * 19 + ch_idx] + 1E-7) + norm_arr[feature_idx, ch_idx, age_idx, 1]

            for feature_idx in range(19 * 13, len(self.feature_name_list)):
                asym_ch1, asym_ch2 = self.feature_name_list[feature_idx].split('_')[2].split('-')
                band_idx = ['Delta', 'Theta', 'Alpha', 'Beta', 'High Beta'].index(self.feature_name_list[feature_idx].split('_')[1])
                ch1_idx, ch2_idx = Util().YB_CH_NAME_LIST.index(asym_ch1), Util().YB_CH_NAME_LIST.index(asym_ch2)

                mean1 = -norm_arr[band_idx, ch1_idx, age_idx, 1] / norm_arr[band_idx, ch1_idx, age_idx, 0]
                sigma1 = 1 / norm_arr[band_idx, ch1_idx, age_idx, 0]
                mean2 = -norm_arr[band_idx, ch2_idx, age_idx, 1] / norm_arr[band_idx, ch2_idx, age_idx, 0]
                sigma2 = 1 / norm_arr[band_idx, ch2_idx, age_idx, 0]

                diff = np.log(self.total_data[age_select, band_idx * 19 + ch1_idx]) - \
                       np.log(self.total_data[age_select, band_idx * 19 + ch2_idx])
                new_mean = mean1 - mean2
                new_sigma = np.sqrt(np.square(sigma1) + np.square(sigma2))
                z_score[age_select, feature_idx] = (diff - new_mean) / new_sigma

        if delete_cz:
            self.total_data = np.delete(self.total_data, np.arange(14, 14 + 19 * 13, 19), axis=1)
            z_score = np.delete(z_score, np.arange(14, 14 + 19 * 13, 19), axis=1)
            self.feature_name_list = np.delete(self.feature_name_list, np.arange(14, 14 + 19 * 13, 19), axis=0)

        z_score = np.where(z_score < -3, -3, np.where(z_score > 3, 3, z_score))
        self.norm_total_data = z_score / 3.

    def age_normalize_dataset(self, delete_cz=True):
        age_bin = [(5, 6), (7, 8), (9, 11), (12, 14), (15, 999)]

        z_score = np.array(self.total_data)

        tmp_norm_data = np.array(self.total_data[np.where(self.total_dz_label[:, 0] == 1)[0]])
        tmp_norm_age = np.array(self.total_age[np.where(self.total_dz_label[:, 0] == 1)[0]])

        for min_age, max_age in age_bin:
            crt_age_idx_norm = np.where((tmp_norm_age >= min_age) & (tmp_norm_age <= max_age))[0]
            crt_age_idx = np.where((self.total_age >= min_age) & (self.total_age <= max_age))[0]
            age_norm_data = tmp_norm_data[crt_age_idx_norm]
            z_score[crt_age_idx, :19 * 13] = (np.log(z_score[crt_age_idx, :19 * 13]) - np.mean(np.log(age_norm_data[:, :19 * 13]), axis=0)) \
                                             / np.std(np.log(age_norm_data[:, :19 * 13]), axis=0)

            z_score[crt_age_idx, 19 * 13:] = (z_score[crt_age_idx, 19 * 13:] - np.mean(age_norm_data[:, 19 * 13:], axis=0)) \
                                             / np.std(age_norm_data[:, 19 * 13:], axis=0)

        if delete_cz:
            self.total_data = np.delete(self.total_data, np.arange(14, 14 + 19 * 13, 19), axis=1)
            z_score = np.delete(z_score, np.arange(14, 14 + 19 * 13, 19), axis=1)
            self.feature_name_list = np.delete(self.feature_name_list, np.arange(14, 14 + 19 * 13, 19), axis=0)

        z_score = np.where(z_score < -3, -3, np.where(z_score > 3, 3, z_score))
        self.norm_total_data = z_score / 3.

    def age_normalize_feature_neuroguide_indiv_data(self, data, age, smooth=True):
        self.age_norm_set = [(0, 9),
                             (9, 12),
                             (12, 15),
                             (15, 20),
                             (20, 30),
                             (30, 80)
                             ]

        age_min = np.max([age - 5, 0])
        z_score = np.zeros_like(data)
        if smooth:
            norm_factor_arr = np.load('./data/ref_file/norm_arr_2.npy')
            for feature_idx in [0, 1, 2, 3, 4, 10, 11, 12]:
                for ch_idx in range(norm_factor_arr.shape[2]):
                    # age_max = np.min([age+5, 0])
                    mean, sigma = np.mean(norm_factor_arr[age_min:age + 5, feature_idx, ch_idx], axis=0)
                    z_score[0, feature_idx * 19 + ch_idx] = (np.log(data[0, feature_idx * 19 + ch_idx]) - mean) / sigma

            for feature_idx in [5, 6, 7, 8, 9]:
                for ch_idx in range(norm_factor_arr.shape[2]):
                    a, b = np.mean(norm_factor_arr[age_min:age + 5, feature_idx, ch_idx], axis=0)
                    z_score[0, feature_idx * 19 + ch_idx] = norm().ppf(beta(a, b).cdf(data[0, feature_idx * 19 + ch_idx] / 100.))

            for feature_idx in range(19 * 13, len(self.feature_name_list)):
                asym_ch1, asym_ch2 = self.feature_name_list[feature_idx].split('_')[2].split('-')
                band_idx = ['Delta', 'Theta', 'Alpha', 'Beta', 'High Beta'].index(self.feature_name_list[feature_idx].split('_')[1])
                ch1_idx, ch2_idx = Util().YB_CH_NAME_LIST.index(asym_ch1), Util().YB_CH_NAME_LIST.index(asym_ch2)
                mean1, sigma1 = np.mean(norm_factor_arr[age_min:age + 5, band_idx, ch1_idx], axis=0)
                mean2, sigma2 = np.mean(norm_factor_arr[age_min:age + 5, band_idx, ch2_idx], axis=0)
                diff = np.log(data[0, band_idx * 19 + ch1_idx]) - np.log(data[0, band_idx * 19 + ch2_idx])
                new_mean = mean1 - mean2
                new_sigma = np.sqrt(np.square(sigma1) + np.square(sigma2))
                z_score[0, feature_idx] = (diff - new_mean) / new_sigma
        else:
            norm_factor_arr = np.load('./data/ref_file/norm_arr_5.npy')
            for feature_idx in [0, 1, 2, 3, 4, 10, 11, 12]:
                # for feature_idx in range(13):
                for ch_idx in range(norm_factor_arr.shape[2]):
                    mean, sigma = np.mean(norm_factor_arr[age:age + 1, feature_idx, ch_idx], axis=0)
                    z_score[0, feature_idx * 19 + ch_idx] = (np.log(data[0, feature_idx * 19 + ch_idx]) - mean) / sigma

            for feature_idx in [5, 6, 7, 8, 9]:
                for ch_idx in range(norm_factor_arr.shape[2]):
                    mean, sigma = np.mean(norm_factor_arr[age:age + 1, feature_idx, ch_idx], axis=0)
                    z_score[0, feature_idx * 19 + ch_idx] = (np.log(data[0, feature_idx * 19 + ch_idx]) - mean) / sigma

            for feature_idx in range(19 * 13, len(self.feature_name_list)):
                asym_ch1, asym_ch2 = self.feature_name_list[feature_idx].split('_')[2].split('-')
                band_idx = ['Delta', 'Theta', 'Alpha', 'Beta', 'High Beta'].index(self.feature_name_list[feature_idx].split('_')[1])
                ch1_idx, ch2_idx = Util().YB_CH_NAME_LIST.index(asym_ch1), Util().YB_CH_NAME_LIST.index(asym_ch2)
                mean1, sigma1 = np.mean(norm_factor_arr[age - 0:age + 1, band_idx, ch1_idx], axis=0)
                mean2, sigma2 = np.mean(norm_factor_arr[age - 0:age + 1, band_idx, ch2_idx], axis=0)
                diff = np.log(data[0, band_idx * 19 + ch1_idx]) - np.log(data[0, band_idx * 19 + ch2_idx])
                new_mean = mean1 - mean2
                new_sigma = np.sqrt(np.square(sigma1) + np.square(sigma2))
                z_score[0, feature_idx] = (diff - new_mean) / new_sigma

        # z_score = np.where(z_score < -5, -5, np.where(z_score > 5, 5, z_score))
        # z_score = z_score / 5.
        return z_score

    def remove_outlier(self, feature_array, age_arr, feature_idx):
        flag_outlier, new_feature_array, new_age_arr = self.detect_outlier(feature_array, age_arr, feature_idx)
        if flag_outlier:
            return self.remove_outlier(new_feature_array, new_age_arr, feature_idx)
        else:
            return new_feature_array, new_age_arr

    def detect_outlier(self, feature_array, age_arr, feature_idx):
        idx_feature_array = feature_array[:, feature_idx]
        gamma_dist, _, _ = Util.get_gamma_dist(idx_feature_array)
        normal_idx = np.where(np.where(gamma_dist.cdf(idx_feature_array) > 0.0005, gamma_dist.cdf(idx_feature_array), 999) < 0.9995)[0]
        if normal_idx.shape[0] == feature_array.shape[0]:
            return False, feature_array, age_arr
        else:
            return True, feature_array[normal_idx], age_arr[normal_idx]

    def outlier_removal_hbn(self):
        new_tbr_arr, new_label_arr = np.array(self.hbn_data), np.array(self.hbn_label)
        for i in range(30):
            for feature_idx in range(6 * 6):
                new_tbr_arr, new_label_arr = self.remove_outlier(new_tbr_arr, new_label_arr, feature_idx)

        self.hbn_data, self.hbn_label = new_tbr_arr, new_label_arr

    def flatten(self, dict_data, y_ch_list):
        feature_list = []
        feature_name_list = []
        for key in dict_data.keys():
            if key in ['abs_power', 'rel_power', 'rat_power']:
                for band_key in dict_data[key].keys():
                    for ch in range(19):
                        feature_list.append(np.array(dict_data[key][band_key])[:, ch])
                        feature_name_list.append('%s_%s_%s' % (key, band_key, y_ch_list[ch]))
            elif key in ['alpha_peak', 'alpha_peak_power']:
                for ch in range(19):
                    feature_list.append(np.array(dict_data[key])[:, ch])
                    feature_name_list.append('%s_%s' % (key, y_ch_list[ch]))
            else:
                feature_list.append(np.array(dict_data[key]))
                feature_name_list.append('%s' % (key))
        return feature_list, feature_name_list

    def load_adhd_data(self):
        fl = os.listdir(self.adhd_data_path)
        for fn in fl:
            if fn.endswith('xlsm'):
                file_name = os.path.join(self.adhd_data_path, fn)
                sheet_1 = pd.read_excel(file_name, 0)
                label_list = {}
                for key in sheet_1.keys():
                    label_list[key] = []
                break

        coh_feature_data_path = '%s/sch_coh' % self.adhd_data_path
        power_feature_data_path = '%s/sch_power' % self.adhd_data_path

        feature_list = []
        label_list = {}

        for json_file in os.listdir(coh_feature_data_path):
            if not json_file.endswith('json'):
                continue
            file_exists = False
            for crt_idx in range(sheet_1['Hospital Number'].shape[0]):
                if sheet_1['Hospital Number'][crt_idx] == int(json_file.split(' ')[0]):
                    file_exists = True
                    break

            if not file_exists:
                print('file does not exist: %s' % json_file)
                continue

            power_feature_data_file_path = os.path.join(power_feature_data_path, "%s_Data_EC_asr_ica.json" % json_file.split(' ')[0])
            coh_feature_data_file_path = os.path.join(coh_feature_data_path, json_file)
            power_feature_val_arr, power_feature_name_list = Util().get_power_feature_from_json(power_feature_data_file_path, False, True, False)
            # coh_feature_val_arr, coh_feature_name_list = Util().get_coh_feature_from_json(coh_feature_data_file_path)
            if power_feature_val_arr is None:
                continue

            mean_feature_val = power_feature_val_arr
            feature_list.append(mean_feature_val)

            for key in sheet_1.keys():
                if label_list.get(key) is None:
                    label_list[key] = []
                label_list[key].append(sheet_1[key][crt_idx])

        self.adhd_data = np.array(feature_list)
        self.adhd_label = label_list
        self.adhd_age = np.where(np.array(self.adhd_label['Age']) < 100, np.array(self.adhd_label['Age']), 17)
        self.where_adhd_adhd = np.where(np.array(self.adhd_label['DZ_G']) == 0)[0] + len(self.where_mipdb) + len(self.where_hbn)
        self.where_adhd_others = np.where(np.array(self.adhd_label['DZ_G']) == 1)[0] + len(self.where_mipdb) + len(self.where_hbn)

        total_dz_label = np.zeros([len(self.adhd_data), self.total_dz_label.shape[1]])
        total_dz_label[:, 1::2] = 1.
        total_dz_label[np.where(np.array(self.adhd_label['DZ_G']) == 0)[0], 36] = 1.
        total_dz_label[np.where(np.array(self.adhd_label['DZ_G']) == 0)[0], 37] = 0.

        if self.total_data is None:
            self.where_adhd = np.arange(self.adhd_data.shape[0])
            self.total_data = self.adhd_data
            self.total_dz_label = self.total_dz_label
            self.total_age = self.adhd_age
        else:
            self.where_adhd = np.arange(self.adhd_data.shape[0]) + self.total_data.shape[0]
            self.total_data = np.concatenate([self.total_data, self.adhd_data], axis=0)
            self.total_dz_label = np.concatenate([self.total_dz_label, total_dz_label], axis=0)
            self.total_age = np.concatenate([self.total_age, self.adhd_age])

        print("SCH data loaded: total number is %d" % self.adhd_data.shape[0])

    def load_hbn_data(self):

        dz_dict = ['No Diagnosis Given', 'Anxiety Disorders', 'Depressive Disorders', 'Trauma and Stressor Related Disorders', 'Disruptive_nan',
                   'Disruptive, Impulse Control and Conduct Disorders', 'Elimination Disorders', 'Obsessive Compulsive and Related Disorders',
                   'Attention-Deficit/Hyperactivity Disorder_ADHD-Combined Type',
                   'Attention-Deficit/Hyperactivity Disorder_ADHD-Hyperactive/Impulsive Type',
                   'Attention-Deficit/Hyperactivity Disorder_ADHD-Inattentive Type',
                   'Attention-Deficit/Hyperactivity Disorder_Other Specified Attention-Deficit/Hyperactivity Disorder',
                   'Attention-Deficit/Hyperactivity Disorder_Unspecified Attention-Deficit/Hyperactivity Disorder',
                   'Autism Spectrum Disorder', 'Specific Learning Disorder', 'Communication Disorder',
                   'Motor Disorder', 'Intellectual Disability']

        label_fn = Path(self.hbn_data_path) / Path('9994_ConsensusDx_20210126.csv')
        import pandas as pds

        label_f = pds.read_csv(label_fn)
        dx_cat_list = []
        for i in range(1, 11):
            dx_cat_list.append(label_f.get('DX_%02d_Cat' % i))

        dx_sub_list = []
        for i in range(1, 11):
            dx_sub_list.append(label_f.get('DX_%02d_Sub' % i))

        dx_type_list = []
        for i in range(1, 11):
            dx_type_list.append(label_f.get('DX_%02d' % i))

        dx_dz_list = []
        for i in range(1, len(dx_sub_list[0])):
            for j in range(len(dx_sub_list)):
                dx_dz_list.append(str(dx_cat_list[j][i]) + '_' + str(dx_sub_list[j][i]) + '_' + str(dx_type_list[j][i]))

        eid = list(label_f.get('EID')[1:])

        feature_file_list = Path(self.hbn_data_path, 'power_feature').iterdir()
        # feature_file_list = os.listdir('%s/power_feature' % self.hbn_data_path)
        c = pd.read_csv(Path(self.hbn_data_path, 'pheno.csv'))

        feature_list = []
        label_list = []
        dz_label_list = []
        file_name_list = np.load(Path(self.hbn_data_path, 'obj_arr.npy'))

        for file_name in file_name_list:
            subject_name = str(file_name).split('/')[4]

            if c.index[c['EID'] == subject_name].__len__() < 1:
                continue

            if subject_name in eid:
                idx_subject = eid.index(subject_name)
            else:
                idx_subject = -1

            age = c['Age'][c.index[c['EID'] == subject_name][-1]]
            sex = c['Sex'][c.index[c['EID'] == subject_name][-1]]
            ehq = c['EHQ_Total'][c.index[c['EID'] == subject_name][-1]]
            if age != age or sex != sex or ehq != ehq:
                continue
            feature_cnt = 0
            for feature_file_name in list(feature_file_list):
                if subject_name not in str(feature_file_name):
                    continue
                power_feature_file_path = Path(self.hbn_data_path, 'power_feature', feature_file_name)
                power_feature_val_arr, power_feature_name_list = Util().get_power_feature_from_json(power_feature_file_path, False, True, False)
                if power_feature_val_arr is None:
                    continue

                if feature_cnt == 0:
                    mean_power_feature_val = np.reshape(np.array(power_feature_val_arr), [-1])
                else:
                    mean_power_feature_val += np.reshape(np.array(power_feature_val_arr), [-1])
                feature_cnt += 1
            if feature_cnt < 1:
                continue
            label_list.append([age, sex, ehq, subject_name])

            tmp_dz_list = np.zeros([len(dz_dict) * 2 + 2])

            if idx_subject != -1:
                for i in range(10):
                    flag_other = True
                    if dx_dz_list[idx_subject * 10 + i] == 'nan_nan_nan':
                        break
                    for dz_dict_idx in range(len(dz_dict)):
                        if dz_dict[dz_dict_idx] in dx_dz_list[idx_subject * 10 + i]:
                            tmp_dz_list[dz_dict_idx * 2] = 1.
                            flag_other = False
                    if flag_other:
                        tmp_dz_list[len(dz_dict) * 2 + 1] = 1.
            else:
                tmp_dz_list[len(dz_dict) * 2] = 1.

            tmp_dz_list[1::2] = np.where(tmp_dz_list[0::2] == 0., 1., 0.)

            mean_feature_val = mean_power_feature_val
            feature_list.append(mean_feature_val / feature_cnt)
            dz_label_list.append(tmp_dz_list)

        self.hbn_data = np.array(feature_list)

        age_child_array = np.array(label_list)[:, 0].astype(np.float32).astype(np.int8)
        self.hbn_age = np.where(age_child_array < 100, age_child_array, 17)

        if self.total_data is None:
            self.where_hbn = np.arange(self.hbn_data.shape[0])
            self.total_data = self.hbn_data
            self.total_age = self.hbn_age
            self.total_dz_label = np.array(dz_label_list)
        else:
            self.where_hbn = np.arange(self.hbn_data.shape[0]) + self.total_data.shape[0]
            self.total_dz_label = np.concatenate([self.total_dz_label, np.array(dz_label_list)], axis=0)
            self.total_data = np.concatenate([self.total_data, self.hbn_data], axis=0)
            self.total_age = np.concatenate([self.total_age, self.hbn_age])

        print("HBN data loaded: total number is %d" % self.hbn_data.shape[0])

    def load_mipdb_data(self):
        feature_file_list = os.listdir('%s/mipdb_feature_power_preprocessed' % self.mipdb_data_path)
        c = pd.read_csv('%s/MIPDB_PublicFile2.csv' % self.mipdb_data_path)

        subject_name_list = []
        for feature_file_name in feature_file_list:
            subject_name_list.append(feature_file_name.split('_')[1][:-3])
        subject_name_list = list(dict.fromkeys(subject_name_list))

        feature_list = []
        label_list = []

        for subject_name in subject_name_list:
            if len(c.index[c['Subject'] == subject_name]) < 1:
                continue
            age = c['Age'][c.index[c['Subject'] == subject_name][-1]]
            if age != age:
                continue
            feature_cnt = 0
            for feature_file_name in feature_file_list:
                if subject_name not in feature_file_name:
                    continue
                power_feature_file_path = os.path.join(self.mipdb_data_path, 'mipdb_feature_power_preprocessed', feature_file_name)
                coh_feature_file_path = os.path.join(self.mipdb_data_path, 'mipdb_coh', feature_file_name)
                power_feature_val_arr, power_feature_name_list = Util().get_power_feature_from_json(power_feature_file_path, False, True, True)
                # coh_feature_val_arr, coh_feature_name_list = Util().get_coh_feature_from_json(coh_feature_file_path)
                if power_feature_val_arr is None:
                    continue

                if feature_cnt == 0:
                    mean_power_feature_val = np.reshape(np.array(power_feature_val_arr), [-1])
                    # mean_coh_feature_val = np.reshape(np.array(coh_feature_val_arr), [-1])
                else:
                    mean_power_feature_val += np.reshape(np.array(power_feature_val_arr), [-1])
                    # mean_coh_feature_val += np.reshape(np.array(coh_feature_val_arr), [-1])
                feature_cnt += 1
            if feature_cnt < 1:
                continue
            label_list.append([age, subject_name])
            mean_feature_val = mean_power_feature_val
            feature_list.append(mean_feature_val / feature_cnt)

        self.mipdb_data = np.array(feature_list)
        age_mipdb_array = np.array(label_list)[:, 0].astype(np.float32).astype(np.int8)
        self.mipdb_age = np.where(age_mipdb_array < 100, age_mipdb_array, 17)

        self.feature_name_list = power_feature_name_list

        total_dz_label = np.zeros([len(self.mipdb_data), self.total_dz_label.shape[1]])
        total_dz_label[:, 1::2] = 1.
        total_dz_label[:, 36] = 1.
        total_dz_label[:, 37] = 0.

        # self.mipdb_data[:, :19 * 6] *= 1E12
        if self.total_data is None:
            self.where_mipdb = np.arange(self.mipdb_data.shape[0])
            self.total_data = self.mipdb_data
            self.total_age = self.mipdb_age
            self.total_dz_label = total_dz_label
        else:
            self.where_mipdb = np.arange(self.mipdb_data.shape[0]) + self.total_data.shape[0]
            self.total_data = np.concatenate([self.total_data, self.mipdb_data], axis=0)
            self.total_dz_label = np.concatenate([self.total_dz_label, total_dz_label], axis=0)
            self.total_age = np.concatenate([self.total_age, self.mipdb_age])

        print("MIPDB data loaded: total number is %d" % self.mipdb_data.shape[0])

    def load_mdd_data(self):
        def load_mdd_data_with_label(label):
            feature_list = []
            sub_list = os.listdir(os.path.join(self.mdd_data_path, label))
            for sub in sub_list:
                if sub == '.DS_Store':
                    continue
                json_file_list = os.listdir(os.path.join(self.mdd_data_path, label, sub))
                subject_feature = []
                for json_file in json_file_list:
                    feature_file = os.path.join(self.mdd_data_path, label, sub, json_file)
                    feature, _ = Util().get_power_feature_from_json(feature_file, False, True, False)
                    if feature is not None:
                        subject_feature.append(feature)
                        # feature_list.append(feature)

                if len(subject_feature) > 0:
                    feature_list.append(np.mean(np.array(subject_feature), axis=0))
            return np.array(feature_list)

        self.mdd_mdd_data = load_mdd_data_with_label('MDD')
        self.mdd_healthy_data = load_mdd_data_with_label('H')
        self.mdd_mdd_age = np.ones([self.mdd_mdd_data.shape[0]]) * 25
        self.mdd_healthy_age = np.ones([self.mdd_healthy_data.shape[0]]) * 25

        if self.total_data is None:
            self.where_mdd = np.arange(self.mdd_mdd_data.shape[0])
            self.where_mdd_mdd = np.arange(self.mdd_mdd_data.shape[0])
            self.where_mdd_healthy = np.arange(self.mdd_healthy_data.shape[0]) + self.mdd_mdd_data.shape[0]
            self.total_data = np.concatenate([self.mdd_mdd_data, self.mdd_healthy_data], axis=0)
            self.total_age = np.concatenate([self.mdd_mdd_age, self.mdd_healthy_age], axis=0)
        else:
            self.where_mdd = np.arange(self.mdd_mdd_data.shape[0] + self.mdd_healthy_data.shape[0]) + self.total_data.shape[0]
            self.where_mdd_mdd = np.arange(self.mdd_mdd_data.shape[0]) + self.total_data.shape[0]
            self.where_mdd_healthy = np.arange(self.mdd_healthy_data.shape[0]) + self.total_data.shape[0] + self.mdd_mdd_data.shape[0]
            self.total_data = np.concatenate([self.total_data, self.mdd_mdd_data, self.mdd_healthy_data], axis=0)
            self.total_age = np.concatenate([self.total_age, self.mdd_mdd_age, self.mdd_healthy_age])

        print("MDD data loaded: total number is %d" % (self.mdd_mdd_data.shape[0] + self.mdd_healthy_data.shape[0]))

    def load_ccss_data(self):
        def load_ccss_data_with_label(label):
            feature_list = []
            sub_list = os.listdir(os.path.join(self.ccss_data_path, label))
            for sub in sub_list:
                if sub == '.DS_Store':
                    continue
                json_file_list = os.listdir(os.path.join(self.ccss_data_path, label, sub))
                for json_file in json_file_list:
                    feature_file = os.path.join(self.ccss_data_path, label, sub, json_file)
                    feature, _ = Util().get_power_feature_from_json(feature_file, False, True, False)
                    if feature is not None:
                        feature_list.append(feature)
            return np.array(feature_list)

        self.ccss_nl_data = load_ccss_data_with_label('NL')
        self.ccss_dep_data = load_ccss_data_with_label('Dep')
        self.ccss_mci_data = load_ccss_data_with_label('MCI')
        self.ccss_c_data = load_ccss_data_with_label('C')
        self.ccss_d_data = load_ccss_data_with_label('D')
        self.ccss_mcd_data = load_ccss_data_with_label('MCD')
        self.ccss_r_data = load_ccss_data_with_label('R')

        self.ccss_nl_age = np.ones([self.ccss_nl_data.shape[0]]) * 40
        self.ccss_dep_age = np.ones([self.ccss_dep_data.shape[0]]) * 40
        self.ccss_mci_age = np.ones([self.ccss_mci_data.shape[0]]) * 40
        self.ccss_c_age = np.ones([self.ccss_c_data.shape[0]]) * 40
        self.ccss_d_age = np.ones([self.ccss_d_data.shape[0]]) * 40
        self.ccss_mcd_age = np.ones([self.ccss_mcd_data.shape[0]]) * 40
        self.ccss_r_age = np.ones([self.ccss_r_data.shape[0]]) * 40

        if self.total_data is None:
            self.where_ccss = np.arange(
                self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0] + self.ccss_mci_data.shape[0] + self.ccss_c_data.shape[0] + self.ccss_d_data.shape[0] +
                self.ccss_mcd_data.shape[0] + self.ccss_r_data.shape[0])
            self.where_ccss_nl = np.arange(self.ccss_nl_data.shape[0])
            self.where_ccss_dep = np.arange(self.ccss_dep_data.shape[0]) + self.ccss_nl_data.shape[0]
            self.where_ccss_mci = np.arange(self.ccss_mci_data.shape[0]) + self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0]
            self.where_ccss_c = np.arange(self.ccss_c_data.shape[0]) + self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0] + self.ccss_mci_data.shape[0]
            self.where_ccss_d = np.arange(self.ccss_d_data.shape[0]) + self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0] + self.ccss_mci_data.shape[0] + \
                                self.ccss_c_data.shape[0]
            self.where_ccss_mcd = np.arange(self.ccss_mcd_data.shape[0]) + self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0] + self.ccss_mci_data.shape[0] + \
                                  self.ccss_c_data.shape[0] + self.ccss_d_data.shape[0]
            self.where_ccss_r = np.arange(self.ccss_r_data.shape[0]) + self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0] + self.ccss_mci_data.shape[0] + \
                                self.ccss_c_data.shape[0] + self.ccss_d_data.shape[0] + self.ccss_mcd_data.shape[0]
            self.total_data = np.concatenate([self.ccss_nl_data, self.ccss_dep_data, self.ccss_mci_data, self.ccss_c_data, self.ccss_d_data, self.ccss_mcd_data, self.ccss_r_data],
                                             axis=0)
            self.total_age = np.concatenate([self.ccss_nl_age, self.ccss_dep_age, self.ccss_mci_age, self.ccss_c_age, self.ccss_d_age, self.ccss_mcd_age, self.ccss_r_age], axis=0)
        else:
            self.where_ccss = np.arange(
                self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0] + self.ccss_mci_data.shape[0] + self.ccss_c_data.shape[0] + self.ccss_d_data.shape[0] +
                self.ccss_mcd_data.shape[0] + self.ccss_r_data.shape[0]) + self.total_data.shape[0]
            self.where_ccss_nl = np.arange(self.ccss_nl_data.shape[0]) + self.total_data.shape[0]
            self.where_ccss_dep = np.arange(self.ccss_dep_data.shape[0]) + self.ccss_nl_data.shape[0] + self.total_data.shape[0]
            self.where_ccss_mci = np.arange(self.ccss_mci_data.shape[0]) + self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0] + self.total_data.shape[0]
            self.where_ccss_c = np.arange(self.ccss_c_data.shape[0]) + self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0] + self.ccss_mci_data.shape[0] + \
                                self.total_data.shape[0]
            self.where_ccss_d = np.arange(self.ccss_d_data.shape[0]) + self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0] + self.ccss_mci_data.shape[0] + \
                                self.ccss_c_data.shape[0] + self.total_data.shape[0]
            self.where_ccss_mcd = np.arange(self.ccss_mcd_data.shape[0]) + self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0] + self.ccss_mci_data.shape[0] + \
                                  self.ccss_c_data.shape[0] + self.ccss_d_data.shape[0] + self.total_data.shape[0]
            self.where_ccss_r = np.arange(self.ccss_r_data.shape[0]) + self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0] + self.ccss_mci_data.shape[0] + \
                                self.ccss_c_data.shape[0] + self.ccss_d_data.shape[0] + self.ccss_mcd_data.shape[0] + self.total_data.shape[0]
            self.total_data = np.concatenate(
                [self.total_data, self.ccss_nl_data, self.ccss_dep_data, self.ccss_mci_data, self.ccss_c_data, self.ccss_d_data, self.ccss_mcd_data, self.ccss_r_data], axis=0)
            self.total_age = np.concatenate(
                [self.total_age, self.ccss_nl_age, self.ccss_dep_age, self.ccss_mci_age, self.ccss_c_age, self.ccss_d_age, self.ccss_mcd_age, self.ccss_r_age], axis=0)

        print("CCSS data loaded: total number is %d" % (
                self.ccss_nl_data.shape[0] + self.ccss_dep_data.shape[0] + self.ccss_mci_data.shape[0] + self.ccss_c_data.shape[0] + self.ccss_d_data.shape[0] +
                self.ccss_mcd_data.shape[0] + self.ccss_r_data.shape[0]))

    def load_data(self):

        self.total_age = []

        delete_cz = True
        if delete_cz:
            ch_num = 18
        else:
            ch_num = 19

        if self.hbn_data_path is not None:
            self.load_hbn_data()

        if self.mipdb_data_path is not None:
            self.load_mipdb_data()

        if self.adhd_data_path is not None:
            self.load_adhd_data()

        self.load_label_data()

        self.age_normalize_dataset(delete_cz=True)

        del_idx_asym = []
        for i in range(5):
            for j in range(12):
                del_idx_asym.append(ch_num * 13 + 20 * i + j + 8)
        for i in range(5):
            del_idx_asym.append(ch_num * 13 + i * 20 + 3)
        for i in range(5):
            del_idx_asym.append(ch_num * 13 + i * 20 + 7)

        self.norm_total_data = np.delete(self.norm_total_data, del_idx_asym, axis=1)
        self.total_data = np.delete(self.total_data, del_idx_asym, axis=1)
        self.feature_name_list = list(np.delete(np.array(self.feature_name_list[:]), del_idx_asym, axis=0))

        # self.norm_total_data = self.norm_total_data[:, ch_num * 5:]
        # self.total_data = self.total_data[:, ch_num * 5:]
        # self.feature_name_list = self.feature_name_list[ch_num * 5:]

        # self.age_normalize_feature_neuroguide(delete_cz=True)

    def load_label_data(self):
        self.age_label = np.zeros([self.total_data.shape[0], 5])
        age_bin = [(5, 6), (7, 8), (9, 11), (12, 14), (15, 999)]
        for i in range(len(age_bin)):
            age_min, age_max = age_bin[i]
            self.age_label[np.where((self.total_age >= age_min) & (self.total_age <= age_max))[0], i] = 1

        self.domain_label = np.zeros([self.total_data.shape[0], 3])
        self.domain_label[self.where_hbn, 0] = 1
        self.domain_label[self.where_mipdb, 1] = 1
        self.domain_label[self.where_adhd, 2] = 1

        self.total_label = self.total_dz_label
        # self.total_label = np.concatenate([self.age_label, self.total_dz_label, self.domain_label], axis=1)

    def age_normalize_feature_norm_data(self, ch_num=19):
        self.age_norm_set = [(0, 9),
                             (9, 12),
                             (12, 15),
                             (15, 20),
                             (20, 30),
                             (30, 80)
                             ]
        norm_idx = list(self.where_hbn_norm) + list(self.where_mipdb)  # + list(self.where_ccss_nl) + list(self.where_mdd_healthy)
        self.norm_total_data = np.zeros_like(self.total_data)

        for feature_idx in range(ch_num * 8):
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 1)
            for age_norm_min, age_norm_max in self.age_norm_set:
                where_norm_age = np.where((self.total_age[norm_idx] >= age_norm_min) & (self.total_age[norm_idx] < age_norm_max))[0]
                tmp_norm_data = np.log(np.array(self.total_data[norm_idx, feature_idx][where_norm_age]))
                where_tmp_age = np.where((self.total_age >= age_norm_min) & (self.total_age < age_norm_max))[0]
                self.norm_total_data[where_tmp_age, feature_idx] = (np.log(self.total_data[where_tmp_age, feature_idx]) - np.mean(tmp_norm_data)) / np.std(tmp_norm_data)

                # ax.plot(np.linspace(age_norm_min, age_norm_max, 20), np.ones(20) * np.exp(np.mean(tmp_norm_data)), color='black')
                # ax.plot(np.linspace(age_norm_min, age_norm_max, 20), np.ones(20) * (np.exp(np.mean(tmp_norm_data) + np.std(tmp_norm_data))), linewidth=0.5, color='red')
                # ax.plot(np.linspace(age_norm_min, age_norm_max, 20), np.ones(20) * (np.exp(np.mean(tmp_norm_data) - np.std(tmp_norm_data))), linewidth=0.5, color='red')
            # fig.savefig('./%s.png' % self.feature_name_list[feature_idx])
            # plt.close(fig)

        for feature_idx in range(ch_num * 8, self.total_data.shape[1]):
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 1)
            for age_norm_min, age_norm_max in self.age_norm_set:
                where_norm_age = np.where((self.total_age[norm_idx] >= age_norm_min) & (self.total_age[norm_idx] < age_norm_max))[0]
                tmp_norm_data = np.array(self.total_data[norm_idx, feature_idx][where_norm_age])
                where_tmp_age = np.where((self.total_age >= age_norm_min) & (self.total_age < age_norm_max))[0]
                self.norm_total_data[where_tmp_age, feature_idx] = (self.total_data[where_tmp_age, feature_idx] - np.mean(tmp_norm_data)) / np.std(tmp_norm_data)

                # ax.plot(np.linspace(age_norm_min, age_norm_max, 20), np.ones(20) * np.mean(tmp_norm_data), color='black')
                # ax.plot(np.linspace(age_norm_min, age_norm_max, 20), np.ones(20) * (np.mean(tmp_norm_data) + np.std(tmp_norm_data)), linewidth=0.5, color='red')
                # ax.plot(np.linspace(age_norm_min, age_norm_max, 20), np.ones(20) * (np.mean(tmp_norm_data) - np.std(tmp_norm_data)), linewidth=0.5, color='red')
            # fig.savefig('./%s.png' % self.feature_name_list[feature_idx])
            # plt.close(fig)

        self.norm_total_data = np.where(self.norm_total_data < -3, -3, np.where(self.norm_total_data > 3, 3, self.norm_total_data))
        self.norm_total_data /= 3

    def age_normalize_feature_norm_indiv_data(self, feature_arr, age, ch_num=19):
        age_norm_set = [(0, 9),
                        (9, 12),
                        (12, 15),
                        (15, 20),
                        (20, 30),
                        (30, 80)]
        norm_idx = list(self.where_hbn_norm) + list(self.where_mipdb) + list(self.where_ccss_nl) + list(self.where_mdd_healthy)
        norm_data = np.zeros_like(feature_arr)

        if age < 9:
            age_norm_min = age_norm_set[0][0]
            age_norm_max = age_norm_set[0][1]
        elif age < 12:
            age_norm_min = age_norm_set[1][0]
            age_norm_max = age_norm_set[1][1]
        elif age < 15:
            age_norm_min = age_norm_set[2][0]
            age_norm_max = age_norm_set[2][1]
        elif age < 20:
            age_norm_min = age_norm_set[3][0]
            age_norm_max = age_norm_set[3][1]
        elif age < 30:
            age_norm_min = age_norm_set[4][0]
            age_norm_max = age_norm_set[4][1]
        else:
            age_norm_min = age_norm_set[5][0]
            age_norm_max = age_norm_set[5][1]

        for feature_idx in range(ch_num * 8):
            where_norm_age = np.where((self.total_age[norm_idx] >= age_norm_min) & (self.total_age[norm_idx] < age_norm_max))[0]
            tmp_norm_data = np.log(np.array(self.total_data[norm_idx, feature_idx][where_norm_age]))
            norm_data[0, feature_idx] = (np.log(feature_arr[0, feature_idx]) - np.mean(tmp_norm_data)) / np.std(tmp_norm_data)

        for feature_idx in range(ch_num * 8, self.total_data.shape[1]):
            where_norm_age = np.where((self.total_age[norm_idx] >= age_norm_min) & (self.total_age[norm_idx] < age_norm_max))[0]
            tmp_norm_data = np.array(self.total_data[norm_idx, feature_idx][where_norm_age])
            norm_data[0, feature_idx] = (feature_arr[0, feature_idx] - np.mean(tmp_norm_data)) / np.std(tmp_norm_data)

        norm_data = np.where(norm_data < -3, -3, np.where(norm_data > 3, 3, norm_data))
        norm_data /= 3
        return norm_data

    def age_normalize_factor_calculate(self):

        def get_gamma_dist(input_list):
            import math
            from scipy.stats import gamma

            N = input_list.__len__()
            tmp_a = 0
            tmp_b = 0
            tmp_c = 0
            tmp_d = 0
            if N == 0:
                k, theta = 0, 0
            else:
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

        con1 = (np.arange(self.total_data.shape[0]) < self.adhd_start_idx)
        con2 = (np.arange(self.total_data.shape[0]) >= self.mipdb_start_idx)
        tmp = []
        for age_idx in range(5, 18):
            con3 = self.total_age == age_idx
            tmp2 = []
            for i in range(self.total_data.shape[1]):
                if 'asymmetry' not in self.feature_name_list[i]:
                    _, k, theta = get_gamma_dist(self.total_data[np.where((con1 | con2) & con3)[0], i])
                    tmp2.append(('gamma', k, theta))
                else:
                    mean = np.mean(self.total_data[np.where((con1 | con2) & con3)[0], i])
                    std = np.std(self.total_data[np.where((con1 | con2) & con3)[0], i])
                    tmp2.append(('normal', mean, std))
            tmp.append(tmp2)

        np.save('./norm_factor_arr.npy', np.array(tmp))
