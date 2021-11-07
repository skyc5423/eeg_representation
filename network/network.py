import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz


def make_network(feature_num, k=8, m=2, beta=0.05, delta=5, gamma=1, noise_std=0.1, kn=5, theta=1, cluster_num=None, pretrain_dir=None, name='network'):
    net = AutoEncoder(name=name, feature_num=feature_num, k=k, cluster_num=cluster_num, m=m, kn=kn, theta=theta, beta=beta, delta=delta, gamma=gamma, noise_std=noise_std,
                      load_weight_path=pretrain_dir)
    return net


class AutoEncoder(torch.nn.Module):
    def __init__(self, name, feature_num, m, beta, delta, gamma, theta, kn, noise_std, cluster_num, fuzzy=True, load_weight_path=None, k=None, node_size_list=None,
                 beta_ae=True):

        super(AutoEncoder, self).__init__()

        if node_size_list is None:
            node_size_list = [256, 256, 256, k, 256, 256, 256, 256]
        if k is None:
            k = node_size_list[3]

        self.k = k
        self.kn = kn
        self.m = m
        self.u = None
        self.beta = beta
        self.beta_ae = beta_ae
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.fuzzy = fuzzy
        self.noise_std = noise_std
        self.load_weight_path = load_weight_path

        self.encoder_1 = torch.nn.Linear(feature_num, node_size_list[0])
        self.encoder_elu_1 = torch.nn.ReLU()
        self.dropout_1 = torch.nn.Dropout(p=0.2)

        self.encoder_2 = torch.nn.Linear(node_size_list[0], node_size_list[1])
        self.encoder_elu_2 = torch.nn.ReLU()
        self.dropout_2 = torch.nn.Dropout(p=0.2)

        self.encoder_3 = torch.nn.Linear(node_size_list[1], node_size_list[2])
        self.encoder_elu_3 = torch.nn.ReLU()
        self.dropout_3 = torch.nn.Dropout(p=0.2)

        self.encoder_4 = torch.nn.Linear(node_size_list[2], 2 * k)

        self.decoder_1 = torch.nn.Linear(k, node_size_list[4])
        self.decoder_elu_1 = torch.nn.ReLU()
        self.decoder_2 = torch.nn.Linear(node_size_list[4], node_size_list[5])
        self.decoder_elu_2 = torch.nn.ReLU()
        # self.decoder_3 = torch.nn.Linear(node_size_list[5], node_size_list[6])
        # self.decoder_elu_3 = torch.nn.ReLU()
        self.decoder_4 = torch.nn.Linear(node_size_list[6], feature_num)
        self.decoder_elu_4 = torch.nn.ReLU()

        self.decoder_out = torch.nn.Linear(feature_num, feature_num)
        self.decoder_tanh_out = torch.nn.Tanh()

        self.discriminator_1 = torch.nn.Linear(feature_num + k, node_size_list[6])
        self.discriminator_relu_1 = torch.nn.ReLU()
        self.discriminator_2 = torch.nn.Linear(node_size_list[6], node_size_list[7])
        self.discriminator_relu_2 = torch.nn.ReLU()
        self.discriminator_out = torch.nn.Linear(node_size_list[7], 1)

        # self.cluster_network_1 = torch.nn.Linear(k, k)
        # self.cluster_network_relu_1 = torch.nn.ReLU()
        # self.cluster_network_2 = torch.nn.Linear(k, k)
        # self.cluster_network_relu_2 = torch.nn.ReLU()
        # self.cluster_network_3 = torch.nn.Linear(k, k)
        self.cluster_network_1 = torch.nn.Linear(k, k * 10)
        self.cluster_network_relu_1 = torch.nn.ReLU()
        self.cluster_network_2 = torch.nn.Linear(k * 10, k * 10)
        self.cluster_network_relu_2 = torch.nn.ReLU()
        if cluster_num is None:
            cluster_num = k
        self.cluster_network_3 = torch.nn.Linear(k * 10, cluster_num)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, eps=1e-07)

        if torch.cuda.is_available():
            self.cuda()
        self.float()

        if self.load_weight_path is not None:
            self.load_weights(self.load_weight_path)

    def save_network_weight(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)

    def __call__(self, input_batch, training=True):

        if torch.cuda.is_available():
            input_batch = torch.from_numpy(input_batch).float().cuda()
        else:
            input_batch = torch.from_numpy(input_batch).float()

        encode_f_batch = self.encode(input_batch)
        encode_f_batch = self.reparameterize(encode_f_batch)
        decode_batch = self.decode(encode_f_batch)

        return encode_f_batch, decode_batch

    def cluster_network(self, encode_f_batch):
        feature_1 = self.cluster_network_relu_1(self.cluster_network_1(encode_f_batch))
        feature_2 = self.cluster_network_relu_2(self.cluster_network_2(feature_1))
        feature_3 = self.cluster_network_3(feature_2)
        return feature_3

    def discriminate(self, input_batch, encode_f_batch):
        input_tmp = torch.cat((input_batch, encode_f_batch), dim=1)
        feature_1 = self.discriminator_relu_1(self.discriminator_1(input_tmp))
        feature_2 = self.discriminator_relu_2(self.discriminator_2(feature_1))
        out = self.discriminator_out(feature_2)
        return out

    def encode(self, input_batch):
        feature_1 = self.dropout_1(self.encoder_elu_1(self.encoder_1(input_batch)))
        feature_2 = self.dropout_2(self.encoder_elu_2(self.encoder_2(feature_1)))
        feature_3 = self.dropout_3(self.encoder_elu_3(self.encoder_3(feature_2)))
        feature_4 = self.encoder_4(feature_3)
        return feature_4

    def decode(self, encode_f_batch):
        feature_1 = self.decoder_elu_1(self.decoder_1(encode_f_batch))
        # feature_2 = self.decoder_elu_2(self.decoder_2(feature_1))
        # feature_3 = self.decoder_elu_3(self.decoder_3(feature_2))
        feature_4 = self.decoder_elu_4(self.decoder_4(feature_1))
        return self.decoder_tanh_out(self.decoder_out(feature_4))

    @staticmethod
    def loss_l2_norm(input_batch, recon_batch):
        return torch.mean(torch.square(input_batch - recon_batch))

    # def diverge_score(self, real_score, fake_score):
    #     real_dis_out = torch.log(2.) - torch.log(1 + torch.exp(-real_score))  # real score increases -> real_dis_out decreases
    #     fake_dis_out = torch.log(2 - torch.exp(torch.log(2.) - torch.log(1 + torch.exp(-fake_score))) + 1E-7)  # fake score increases -> fake_dis_out decreases
    #     return torch.sigmoid(real_dis_out), torch.sigmoid(fake_dis_out)

    def loss_encoder_cluster(self, input_batch, encode_f_real, encode_f_fake, y_cluster):
        if y_cluster is not None:
            log_var_y = torch.log(torch.square(torch.std(y_cluster, dim=0)))
            mu_y = torch.mean(y_cluster - torch.mean(y_cluster, dim=0), dim=0)
            loss_kld_y = torch.mean(-0.5 * (1 + log_var_y - mu_y ** 2 - log_var_y.exp()))
        else:
            loss_kld_y = 0

        mu_f, log_var_f = torch.split(encode_f_real, [self.k, self.k], dim=1)
        loss_kld_f = torch.mean(-0.5 * torch.sum(1 + log_var_f - mu_f ** 2 - log_var_f.exp(), dim=1), dim=0)

        reparam_f_real = self.reparameterize(encode_f_real)
        reparam_f_fake = self.reparameterize(encode_f_fake)

        discriminator_out_real = self.discriminate(input_batch, reparam_f_real)
        discriminator_out_fake = self.discriminate(input_batch, reparam_f_fake)

        real_dis_out = torch.sigmoid(discriminator_out_real)
        fake_dis_out = torch.sigmoid(discriminator_out_fake)

        loss_discriminator = torch.mean(- (torch.log(real_dis_out + 1E-7) + torch.log(1 - fake_dis_out + 1E-7)))
        return self.gamma * loss_kld_f + loss_kld_y + self.beta * loss_discriminator

    def loss_decoder(self, input_batch, decode_batch, decode_batch_noise=None):
        # return self.delta * torch.mean(torch.sqrt(torch.square(input_batch - decode_batch)))
        if decode_batch_noise is None:
            return self.delta * torch.mean(torch.pow(input_batch - decode_batch, 2))
        else:
            return self.delta * torch.mean(torch.pow(input_batch - decode_batch, 2)) + self.delta * torch.mean(torch.pow(decode_batch_noise - decode_batch, 2))

    def affinity_mat_fuzzy(self, input_feature):
        input_feature = torch.from_numpy(input_feature)
        zi = torch.unsqueeze(input_feature, 0).repeat(input_feature.shape[0], 1, 1)
        zj = torch.unsqueeze(input_feature, 1).repeat(1, input_feature.shape[0], 1)
        w = torch.mean(torch.exp(-torch.square(zi - zj) / 0.01), dim=2)
        return w

    def affinity_mat(self, encode_f_real):
        zi = torch.unsqueeze(encode_f_real, 0).repeat(encode_f_real.shape[0], 1, 1)
        zj = torch.unsqueeze(encode_f_real, 1).repeat(1, encode_f_real.shape[0], 1)
        w = torch.mean(torch.exp(-torch.square(zi - zj) / 1), dim=2)
        return w

    def loss_cluster(self, w, y):
        if y is None:
            return 0
        yi = torch.unsqueeze(y, 0).repeat([y.shape[0], 1, 1])
        yj = torch.unsqueeze(y, 1).repeat([1, y.shape[0], 1])
        yij = torch.square(yi - yj)

        if torch.cuda.is_available():
            return torch.mean(torch.sum(torch.mul(torch.unsqueeze(w.cuda(), 2), yij), dim=2))
            # return torch.mean(torch.mul(w.cuda(), torch.sqrt(torch.sum(yij, dim=2))))
        else:
            return torch.mean(torch.mul(w, torch.sqrt(torch.sum(yij, dim=2))))

    def compute_cholesky_if_possible(self, cluster_y_tilde):
        try:
            cholesky_l = torch.cholesky(torch.matmul(torch.transpose(cluster_y_tilde, 0, 1), cluster_y_tilde))
            return cholesky_l
        except:
            jitter = 1E-9
            while jitter < 2.0:
                try:
                    cholesky_l = torch.cholesky(torch.matmul(torch.transpose(cluster_y_tilde, 0, 1), cluster_y_tilde) + jitter * torch.eye(cluster_y_tilde.shape[1]))
                    return cholesky_l
                except:
                    jitter *= 1.1

            return

    def compute_cluster_y(self, input_batch):
        input_batch = torch.from_numpy(input_batch).float()
        encode_f_real = torch.split(self.encode(input_batch), [self.k, self.k], dim=1)[0]
        # encode_f_real = self.reparameterize(self.encode(input_batch))
        cluster_y_tilde = self.cluster_network(encode_f_real)  # M x D
        cholesky_l = self.compute_cholesky_if_possible(cluster_y_tilde)
        cluster_y = torch.matmul(cluster_y_tilde, torch.transpose(torch.inverse(cholesky_l), 0, 1)) * np.sqrt(encode_f_real.shape[0])
        return cluster_y

    def fuzzy_spectral_cluster(self, input_feature):
        if torch.cuda.is_available():
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(input_feature.cpu().detach().numpy().T, 6, self.m, error=0.005, maxiter=1000, init=None)
        else:
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(input_feature.detach().numpy().T, 6, self.m, error=0.005, maxiter=1000, init=None)

        self.v_array = cntr
        self.u = u
        self.d = d

    def set_encoder_trainable(self):
        for n, p in self.named_parameters():
            if n.startswith('decoder'):
                p.requires_grad = False
            else:
                p.requires_grad = True

    def set_decoder_trainable(self):
        for n, p in self.named_parameters():
            if n.startswith('encoder'):
                p.requires_grad = False
            else:
                p.requires_grad = True

    def set_all_trainable(self):
        for n, p in self.named_parameters():
            p.requires_grad = True

    def mutual_info_criterion(self, encode_f_real, reparam_f_real):
        mu_f, log_var_f = torch.split(encode_f_real, [self.k, self.k], dim=1)
        total_mu_f = torch.mean(reparam_f_real, dim=0)
        total_log_var_f = torch.log(torch.std(reparam_f_real, dim=0) + 1E-7)
        mutual_info_1 = torch.mean(torch.sum(1 + log_var_f - mu_f ** 2 - log_var_f.exp(), dim=1), dim=0)
        mutual_info_2 = - torch.mean(1 + total_log_var_f - total_mu_f ** 2 - total_log_var_f.exp(), dim=0)
        return mutual_info_1 - mutual_info_2

    def pretrain_vae(self, input_batch, aggressive):
        if aggressive:
            self.set_encoder_trainable()
            original_loss = 9999.
            while True:
                if not isinstance(input_batch, np.ndarray):
                    input_batch = input_batch.cpu().detach().numpy()
                fake_batch = np.array(input_batch)
                np.random.shuffle(fake_batch)

                if torch.cuda.is_available():
                    input_batch = torch.from_numpy(input_batch).cuda()
                    fake_batch = torch.from_numpy(fake_batch).cuda()
                else:
                    input_batch = torch.from_numpy(input_batch)
                    fake_batch = torch.from_numpy(fake_batch)

                input_batch = input_batch.float()
                fake_batch = fake_batch.float()

                encode_f_real = self.encode(input_batch)
                encode_f_fake = self.encode(fake_batch)

                reparam_f_real = self.reparameterize(encode_f_real)

                loss_encoder = self.loss_encoder_cluster(input_batch, encode_f_real, encode_f_fake, None)

                if torch.cuda.is_available():
                    norm = torch.tensor([0], dtype=torch.float).cuda()
                else:
                    norm = torch.tensor([0], dtype=torch.float)
                for parameter in self.parameters():
                    # norm += torch.norm(parameter, p=1)
                    norm += torch.norm(parameter, p=2)

                # decode_img_noise = self.decode(noised_encode_f_real)
                decode_batch = self.decode(reparam_f_real)

                loss_decoder = self.loss_decoder(input_batch, decode_batch)

                loss_total = loss_encoder + loss_decoder  # + 0.0002 * norm
                # print("Here 1: %f, %f" % (loss_encoder, loss_decoder))
                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                encode_f_real_tmp = self.encode(input_batch)
                # if torch.mean(encode_f_real_tmp).detach().numpy() != torch.mean(encode_f_real_tmp).detach().numpy():
                #     print()

                if original_loss - loss_total.cpu().detach().numpy() < 1E-3:
                    break
                original_loss = loss_total.cpu().detach().numpy()

            self.set_decoder_trainable()
            if not isinstance(input_batch, np.ndarray):
                input_batch = input_batch.cpu().detach().numpy()
            fake_batch = np.array(input_batch)
            np.random.shuffle(fake_batch)

            if torch.cuda.is_available():
                input_batch = torch.from_numpy(input_batch).cuda()
                fake_batch = torch.from_numpy(fake_batch).cuda()
            else:
                input_batch = torch.from_numpy(input_batch)
                fake_batch = torch.from_numpy(fake_batch)

            input_batch = input_batch.float()
            fake_batch = fake_batch.float()

            encode_f_real = self.encode(input_batch)
            encode_f_fake = self.encode(fake_batch)

            loss_encoder = self.loss_encoder_cluster(input_batch, encode_f_real, encode_f_fake, None)

            reparam_f_real = self.reparameterize(encode_f_real)

            if torch.cuda.is_available():
                norm = torch.tensor([0], dtype=torch.float).cuda()
            else:
                norm = torch.tensor([0], dtype=torch.float)
            for parameter in self.parameters():
                # norm += torch.norm(parameter, p=1)
                norm += torch.norm(parameter, p=2)

            # decode_img_noise = self.decode(noised_encode_f_real)
            decode_batch = self.decode(reparam_f_real)

            loss_decoder = self.loss_decoder(input_batch, decode_batch)

            loss_total = loss_encoder + loss_decoder  # + 0.0002 * norm

            # print("Here 2: %f, %f" % (loss_encoder, loss_decoder))
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            encode_f_real_tmp = self.encode(input_batch)
            # if torch.mean(encode_f_real_tmp).detach().numpy() != torch.mean(encode_f_real_tmp).detach().numpy():
            #     print()
            reparam_f_real_tmp = self.reparameterize(encode_f_real_tmp)
            mutual_info = self.mutual_info_criterion(encode_f_real_tmp, reparam_f_real_tmp)

            return loss_total, loss_encoder, loss_decoder, 0.0002 * norm, mutual_info.cpu().detach().numpy()
        else:

            self.set_all_trainable()
            if not isinstance(input_batch, np.ndarray):
                input_batch = input_batch.cpu().detach().numpy()
            fake_batch = np.array(input_batch)
            np.random.shuffle(fake_batch)

            if torch.cuda.is_available():
                input_batch = torch.from_numpy(input_batch).cuda()
                fake_batch = torch.from_numpy(fake_batch).cuda()
            else:
                input_batch = torch.from_numpy(input_batch)
                fake_batch = torch.from_numpy(fake_batch)

            input_batch = input_batch.float()
            fake_batch = fake_batch.float()

            encode_f_real = self.encode(input_batch)
            encode_f_fake = self.encode(fake_batch)

            loss_encoder = self.loss_encoder_cluster(input_batch, encode_f_real, encode_f_fake, None)

            reparam_f_real = self.reparameterize(encode_f_real)

            if torch.cuda.is_available():
                norm = torch.tensor([0], dtype=torch.float).cuda()
            else:
                norm = torch.tensor([0], dtype=torch.float)
            for parameter in self.parameters():
                # norm += torch.norm(parameter, p=1)
                norm += torch.norm(parameter, p=2)

            # decode_img_noise = self.decode(noised_encode_f_real)
            decode_batch = self.decode(reparam_f_real)

            loss_decoder = self.loss_decoder(input_batch, decode_batch)

            loss_total = loss_encoder + loss_decoder  # + 0.0002 * norm

            # print("Here 3: %f, %f" % (loss_encoder, loss_decoder))
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            encode_f_real_tmp = self.encode(input_batch)
            reparam_f_real_tmp = self.reparameterize(encode_f_real_tmp)
            mutual_info = self.mutual_info_criterion(encode_f_real_tmp, reparam_f_real_tmp)

            # encode_f_real = self.encode(input_batch)
            # if torch.mean(encode_f_real_tmp).detach().numpy() != torch.mean(encode_f_real_tmp).detach().numpy():
            #     print()

            return loss_total, loss_encoder, loss_decoder, 0.0002 * norm, mutual_info.cpu().detach().numpy()

    def reparameterize(self, encode_f):
        mu, log_var = torch.split(encode_f, [self.k, self.k], dim=1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def train_autoencoder(self, input_image, cluster=True):
        fake_image = np.array(input_image)
        np.random.shuffle(fake_image)

        if torch.cuda.is_available():
            input_image = torch.from_numpy(input_image).cuda()
            fake_image = torch.from_numpy(fake_image).cuda()
        else:
            input_image = torch.from_numpy(input_image)
            fake_image = torch.from_numpy(fake_image)

        input_image = input_image.float()
        fake_image = fake_image.float()

        encode_f_real = self.encode(input_image)
        encode_f_real_reparam = self.reparameterize(encode_f_real)
        encode_f_fake = self.encode(fake_image)

        if torch.cuda.is_available():
            noised_encode_f_real = torch.normal(1, self.noise_std, encode_f_real_reparam.shape).cuda() * encode_f_real_reparam
        else:
            noised_encode_f_real = torch.normal(1, self.noise_std, encode_f_real_reparam.shape) * encode_f_real_reparam

        decode_img_noise = self.decode(noised_encode_f_real)
        decode_img_real = self.decode(encode_f_real_reparam)

        loss_decoder = self.loss_decoder(input_image, decode_img_real, decode_img_noise)
        #
        # norm = torch.FloatTensor([0])
        # for parameter in self.parameters():
        #     norm += torch.norm(parameter, p=1)
        #     norm += torch.norm(parameter, p=2)

        cluster_y_tilde = self.cluster_network(encode_f_real_reparam)  # M x D

        # if False:
        # self.fuzzy_spectral_cluster(encode_f_real_reparam)
        # affinity_mat = self.affinity_mat_fuzzy(np.power(np.e, - self.d.T))  # M x M x D
        # else:
        affinity_mat = self.affinity_mat(encode_f_real_reparam)

        if True:
            tmp = torch.where(
                affinity_mat[:, :] - torch.eye(encode_f_real_reparam.shape[0]).cuda() -
                torch.topk(affinity_mat - torch.eye(encode_f_real_reparam.shape[0]).cuda(), self.kn + 1)[0][:, -1] > 0, torch.tensor(1).cuda(), torch.tensor(0).cuda())
            tmp = tmp + tmp.transpose(1, 0)
            affinity_mat = torch.mul(affinity_mat, torch.where(tmp > 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()))

        cholesky_l = self.compute_cholesky_if_possible(cluster_y_tilde)
        if cholesky_l is None:
            cluster_y = None
        else:
            cluster_y = torch.matmul(cluster_y_tilde, torch.transpose(torch.inverse(cholesky_l), 0, 1)) * np.sqrt(encode_f_real.shape[0])
        loss_encoder = 1. * self.loss_encoder_cluster(input_image, encode_f_real, encode_f_fake, cluster_y)
        loss_cluster = self.theta * self.loss_cluster(affinity_mat, cluster_y)

        loss_total = loss_encoder + loss_decoder + loss_cluster

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        return loss_total, loss_encoder, loss_decoder, loss_cluster
