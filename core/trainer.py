# from deep_neural_net import Network
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, network, db_helper, output_directory: str, scan_network=None):
        self.network = network
        self.db_helper = db_helper
        self.scan_network = scan_network
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)

    def pretrain_network(self, max_epoch=50000, save_result_per_epoch: int = None):
        feature_array = self.db_helper.norm_total_data
        save_flag = True
        loss_list = []
        aggressive = True
        original_mi = -999
        self.network.train()
        for epoch_idx in range(1, max_epoch + 1):
            # if epoch_idx < 10000:
            #     self.network.gamma = 0
            # else:
            #     self.network.gamma = 1.
            idx = np.arange(feature_array.shape[0])
            np.random.shuffle(idx)
            for iter_idx in range(1):
                random_idx = idx[iter_idx * int(idx.shape[0] / 1):(iter_idx + 1) * int(idx.shape[0] / 1)]
                loss_tmp, loss_encoder, loss_decoder, loss_norm, mi = self.network.pretrain_vae(feature_array[random_idx], aggressive=aggressive)
                print("%.6f - %.6f = %.6f" % (mi, original_mi, mi - original_mi))
                if mi - original_mi < 1E-6:
                    original_mi = mi
                    aggressive = False
                else:
                    aggressive = True
                    original_mi = mi
                if epoch_idx % 10 == 0:
                    loss_list.append([loss_tmp, loss_encoder, loss_decoder, loss_norm])
            print("Pretrain %02d: total=%f, encoder=%f, decoder=%f, norm=%f" % (epoch_idx, loss_tmp, loss_encoder, loss_decoder, loss_norm))

            if save_result_per_epoch is not None:
                if epoch_idx % save_result_per_epoch == 0:
                    if not os.path.isdir('./%s/pretrain_result/' % self.output_directory):
                        os.mkdir('./%s/pretrain_result/' % self.output_directory)
                    fig, ax = plt.subplots(4, 4)
                    for i in range(4):
                        for j in range(4):
                            ax[i, j].plot(feature_array[random_idx[4 * i + j]], color='blue', linewidth=0.3)
                            tmp = self.network(feature_array[random_idx[4 * i + j:4 * i + j + 1]])
                            ax[i, j].plot(tmp[1][0].cpu().detach().numpy(), color='red', linewidth=0.3)
                            ax[i, j].set_ylim(-1, 1)
                    fig.savefig('./%s/pretrain_result/tmp_%d.png' % (self.output_directory, epoch_idx))
                    plt.close(fig)

                    fig, ax = plt.subplots(1, 4, figsize=(15, 6))
                    for i in range(4):
                        ax[i].plot(np.array(loss_list)[:, i], 'black')
                    fig.savefig('./%s/pretrain_result/loss_pretrain.png' % (self.output_directory))
                    plt.close(fig)

                    self.network.save_network_weight('./%s/pre_trained_network_at_%d' % (self.output_directory, epoch_idx))

        self.network.save_network_weight('./%s/pre_trained_network' % self.output_directory)

    def train_network(self, pre_trained_network_epoch: int = None, max_epoch=50000, save_result_per_epoch: int = None, save_directory: str = None):

        if save_result_per_epoch is None:
            print('Argument save_result_per_epoch should be determined')
        if save_directory is None:
            print('Argument save_directory should be determined')

        # if pre_trained_network_epoch is None:
        self.network.load_weights('./%s/%s' % (self.output_directory, 'pre_trained_network'))
        # else:
        #     self.network.load_weights('./%s/%s' % (self.output_directory, 'pre_trained_network_at_%d' % pre_trained_network_epoch))

        if not os.path.isdir('./%s/%s' % (self.output_directory, save_directory)):
            os.mkdir('./%s/%s' % (self.output_directory, save_directory))

        if not os.path.isdir('./%s/%s/result_figure' % (self.output_directory, save_directory)):
            os.mkdir('./%s/%s/result_figure' % (self.output_directory, save_directory))

        loss_total_list = []
        loss_encoder_list = []
        loss_decoder_list = []
        loss_cluster_list = []
        loss_norm_list = []

        feature_array = self.db_helper.norm_total_data

        for epoch_idx in range(1, max_epoch + 1):
            idx = np.arange(feature_array.shape[0])
            np.random.shuffle(idx)
            for iter_idx in range(1):
                random_idx = idx[iter_idx * int(idx.shape[0] / 1):(iter_idx + 1) * int(idx.shape[0] / 1)]
                loss_tmp, loss_encoder, loss_decoder, loss_cluster, loss_norm = self.network.train_(feature_array[random_idx], epoch_idx)
            print(
                "Cluster Train %02d: total=%f, encoder=%f, decoder=%f, cluster=%f, norm=%f" % (epoch_idx, loss_tmp, loss_encoder, loss_decoder, loss_cluster, loss_norm))

            if epoch_idx % 10 == 0:
                loss_total_list.append(loss_tmp)
                loss_encoder_list.append(loss_encoder)
                loss_decoder_list.append(loss_decoder)
                loss_cluster_list.append(loss_cluster)
                loss_norm_list.append(loss_norm)

            if epoch_idx % save_result_per_epoch == 0:
                self.network.save_network_weight('./%s/%s/trained_network_%d' % (self.output_directory, save_directory, epoch_idx))
                fig, ax = plt.subplots(5, 1, figsize=(6, 12))
                ax[0].plot(10 * np.arange(len(loss_total_list)), loss_total_list)
                ax[1].plot(10 * np.arange(len(loss_total_list)), loss_encoder_list)
                ax[2].plot(10 * np.arange(len(loss_total_list)), loss_decoder_list)
                ax[3].plot(10 * np.arange(len(loss_total_list)), loss_cluster_list)
                ax[4].plot(10 * np.arange(len(loss_total_list)), loss_norm_list)
                fig.savefig('./%s/%s/result_figure/loss.png' % (self.output_directory, save_directory))
                plt.close(fig)
                fig, ax = plt.subplots(4, 4)
                for i in range(4):
                    for j in range(4):
                        ax[i, j].plot(feature_array[random_idx[4 * i + j]], color='blue', linewidth=0.3)
                        if torch.cuda.is_available():
                            ax[i, j].plot(self.network(feature_array[random_idx[4 * i + j:4 * i + j + 1]])[1][0].cpu().detach().numpy(), color='red', linewidth=0.3)
                        else:
                            ax[i, j].plot(self.network(feature_array[random_idx[4 * i + j:4 * i + j + 1]])[1][0].detach().numpy(), color='red', linewidth=0.3)

                fig.savefig('./%s/%s/result_figure/decoded_%d.png' % (self.output_directory, save_directory, epoch_idx))
                plt.close(fig)

    def train_scan(self, pre_train_network, max_epoch=50000, save_result_per_epoch: int = None, save_directory: str = None):

        if save_result_per_epoch is None:
            print('Argument save_result_per_epoch should be determined')
        if save_directory is None:
            print('Argument save_directory should be determined')

        if not os.path.isdir('./%s' % (self.output_directory)):
            os.mkdir('./%s' % (self.output_directory))

        if not os.path.isdir('./%s/result_figure' % (self.output_directory)):
            os.mkdir('./%s/result_figure' % (self.output_directory))

        loss_total_list = []
        loss_encoder_list = []
        loss_decoder_age_list = []
        loss_decoder_dz_list = []
        loss_decoder_site_list = []
        loss_kld_scan_list = []
        loss_norm_list = []

        feature_array = self.db_helper.norm_total_data
        label_array = self.db_helper.total_label

        for epoch_idx in range(1, max_epoch + 1):
            idx = np.arange(feature_array.shape[0])
            np.random.shuffle(idx)
            for iter_idx in range(1):
                random_idx = idx[iter_idx * int(idx.shape[0] / 1):(iter_idx + 1) * int(idx.shape[0] / 1)]
                loss_total, loss_encoder, loss_decoder, loss_kld_scan_bvae, loss_norm = self.scan_network.train_scan(feature_array[random_idx], label_array[random_idx],
                                                                                                                     pre_train_network)
            print(
                "Cluster Train %02d: total=%f, encoder=%f, decoder=%f, kld_scan=%f, norm=%f" % (
                    epoch_idx, loss_total, loss_encoder, loss_decoder, loss_kld_scan_bvae, loss_norm))
            print(
                "Cluster Train %02d: dz=%f" % (
                    epoch_idx, loss_decoder))
            # print(
            #     "Cluster Train %02d: age=%f, dz=%f, site=%f" % (
            #         epoch_idx, loss_decoder[0], loss_decoder[1], loss_decoder[2]))

            if epoch_idx % 10 == 0:
                loss_total_list.append(loss_total)
                loss_encoder_list.append(loss_encoder)
                # loss_decoder_age_list.append(loss_decoder[0])
                loss_decoder_dz_list.append(loss_decoder)
                # loss_decoder_site_list.append(loss_decoder[2])
                loss_kld_scan_list.append(loss_kld_scan_bvae)
                loss_norm_list.append(loss_norm)

            if epoch_idx % save_result_per_epoch == 0:
                self.scan_network.save_network_weight('./%s/trained_network_%d' % (self.output_directory, epoch_idx))
                fig, ax = plt.subplots(5, 1, figsize=(6, 12))
                ax[0].plot(10 * np.arange(len(loss_total_list)), loss_total_list)
                ax[1].plot(10 * np.arange(len(loss_total_list)), loss_encoder_list)
                # ax[2].plot(10 * np.arange(len(loss_total_list)), loss_decoder_age_list, 'r')
                ax[2].plot(10 * np.arange(len(loss_total_list)), loss_decoder_dz_list)
                # ax[2].plot(10 * np.arange(len(loss_total_list)), loss_decoder_site_list, 'k')
                ax[3].plot(10 * np.arange(len(loss_total_list)), loss_kld_scan_list)
                ax[4].plot(10 * np.arange(len(loss_total_list)), loss_norm_list)
                fig.savefig('./%s/result_figure/loss.png' % (self.output_directory))
                plt.close(fig)

                fig, ax = plt.subplots(4, 2)
                if torch.cuda.is_available():
                    en_label = self.network.encode(torch.from_numpy(self.db_helper.norm_total_data[random_idx[:8]]).float().cuda())[:, :self.network.k]
                else:
                    en_label = self.network.encode(torch.from_numpy(self.db_helper.norm_total_data[random_idx[:8]]).float())[:, :self.network.k]
                de_label = self.scan_network.decode_label(en_label)
                for n in range(8):
                    for i in range(19):
                        sm = torch.softmax(de_label[n, i * 2:i * 2 + 2], dim=0)
                        if torch.cuda.is_available():
                            ax[int(n / 2), n % 2].bar(np.arange(2) + 6 * i, sm.cpu().detach().numpy(), color='red')
                        else:
                            ax[int(n / 2), n % 2].bar(np.arange(2) + 6 * i, sm.detach().numpy(), color='red')
                    ax[int(n / 2), n % 2].bar(np.arange(19) * 6, self.db_helper.total_label[n, 0::2], color='blue', alpha=0.5)
                    ax[int(n / 2), n % 2].bar(np.arange(19) * 6 + 1, self.db_helper.total_label[n, 1::2], color='green', alpha=0.5)
                fig.savefig('./%s/result_figure/res.png' % (self.output_directory))
                plt.close(fig)
                # fig, ax = plt.subplots(4, 4)
                # for i in range(4):
                #     for j in range(4):
                #         ax[i, j].plot(feature_array[random_idx[4 * i + j]], color='blue', linewidth=0.3)
                #         if torch.cuda.is_available():
                #             ax[i, j].plot(self.network(feature_array[random_idx[4 * i + j:4 * i + j + 1]])[1][0].cpu().detach().numpy(), color='red', linewidth=0.3)
                #         else:
                #             ax[i, j].plot(self.network(feature_array[random_idx[4 * i + j:4 * i + j + 1]])[1][0].detach().numpy(), color='red', linewidth=0.3)
                #
                # fig.savefig('./%s/%s/result_figure/decoded_%d.png' % (self.output_directory, save_directory, epoch_idx))
                # plt.close(fig)
