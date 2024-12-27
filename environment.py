import numpy as np
import math

class RIS_MISO(object):
    def __init__(self,
                 num_antennas,
                 num_RIS_elements,
                 num_users_R,
                 num_users_T,
                 channel_est_error=False,
                 AWGN_var=1e-2,
                 channel_noise_var=1e-2):

        self.power_unit = 100  # 最大发送功率，单位mw
        self.k_B = 0.01
        self.k_S = 0.01

        self.M = num_antennas  # 基站天线数
        self.N = num_RIS_elements  # STAR单元数
        self.KR = num_users_R  # 反射用户数量
        self.KT = num_users_T  # 透射用户数量
        self.K = self.KR + self.KT  # 总用户数

        self.channel_est_error = channel_est_error

        assert self.M == self.K  # 基站天线数M必须等于总用户数K

        self.awgn_var = AWGN_var  # 用户处的AWGN的功率
        self.channel_noise_var = channel_noise_var  # 信道噪声功率

        power_size = 2 * self.K

        channel_size = 2 * self.N * self.K + 2 * self.M * self.N

        self.action_dim = 2 * self.M * self.K + 4 * self.N
        self.state_dim = power_size + channel_size + self.action_dim

        self.H_R_KR = None  # STAR-RIS-反射用户
        self.H_R_KT = None  # STAR-RIS-透射用户
        self.H_B_R = None   # 基站-STAR-RIS

        self.G = np.ones(shape=(self.M, self.K), dtype=complex)

        self.Phi_R = np.eye(self.N, dtype=complex)
        self.Phi_T = np.eye(self.N, dtype=complex)

        self.data_rate_list_R = np.zeros(self.KR)
        self.data_rate_list_T = np.zeros(self.KT)

        self.state = None
        self.done = None

        self.episode_t = None

    def _compute_H_R_K_tilde(self):

        A = self.H_R_KR.T @ self.Phi_R @ self.H_B_R @ self.G
        B = self.H_R_KT.T @ self.Phi_T @ self.H_B_R @ self.G

        return A, B

    def reset(self):
        self.episode_t = 0

        # 瑞丽信道矩阵

        self.H_R_KR = np.random.normal(0, np.sqrt(0.5), (self.N, self.KR)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.N, self.KR))
        self.H_R_KT = np.random.normal(0, np.sqrt(0.5), (self.N, self.KT)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.N, self.KT))
        self.H_B_R = np.random.normal(0, np.sqrt(0.5), (self.N, self.M)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.N, self.M))

        # 行向量化

        init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), np.imag(self.G.reshape(1, -1))))
        init_action_Phi_R = np.hstack(
            (np.real(np.diag(self.Phi_R)).reshape(1, -1), np.imag(np.diag(self.Phi_R)).reshape(1, -1)))
        init_action_Phi_T = np.hstack(
            (np.real(np.diag(self.Phi_T)).reshape(1, -1), np.imag(np.diag(self.Phi_T)).reshape(1, -1)))

        init_action = np.hstack((init_action_G, init_action_Phi_R, init_action_Phi_T))

        Phi_R_real = init_action[:, -4 * self.N:-3 * self.N]
        Phi_R_imag = init_action[:, -3 * self.N:-2 * self.N]

        Phi_T_real = init_action[:, -2 * self.N:-self.N]
        Phi_T_imag = init_action[:, -self.N:]

        self.Phi_R = np.eye(self.N, dtype=complex) * (Phi_R_real + 1j * Phi_R_imag)
        self.Phi_T = np.eye(self.N, dtype=complex) * (Phi_T_real + 1j * Phi_T_imag)

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        H_R_K_tilde_1, H_R_K_tilde_2 = self._compute_H_R_K_tilde()
        H_R_K_array = np.concatenate((H_R_K_tilde_1, H_R_K_tilde_2), axis=0)

        power_r = np.linalg.norm(H_R_K_array, axis=0).reshape(1, -1) ** 2  # k维的行向量

        # 信道矩阵实部虚部分开

        H_R_KR_real, H_R_KR_imag = np.real(self.H_R_KR).reshape(1, -1), np.imag(self.H_R_KR).reshape(1, -1)
        H_R_KT_real, H_R_KT_imag = np.real(self.H_R_KT).reshape(1, -1), np.imag(self.H_R_KT).reshape(1, -1)
        H_B_R_real, H_B_R_imag = np.real(self.H_B_R).reshape(1, -1), np.imag(self.H_B_R).reshape(1, -1)

        self.state = np.hstack((init_action, power_t, power_r, H_R_KR_real, H_R_KR_imag, H_R_KT_real, H_R_KT_imag, H_B_R_real, H_B_R_imag))

        return self.state

    def _compute_reward(self, Phi_R, Phi_T):

        reward = 0
        opt_reward = 0
        beam_sum = np.zeros(shape=(self.K, self.K), dtype=complex)

        for user_k in range(self.K):
            beam_sum += np.dot(self.G[:, user_k], self.G[:, user_k].conj().T)  # 维度：M*M

        diag_tilde = np.diag(np.diag(beam_sum))

        for k_R in range(self.KR):
            # Computing cascaded channels
            H_R_KR_H = self.H_R_KR[:, k_R].conj().T  # 先取出单个用户的信道向量的共轭转置 维度：1*N
            h_mid = np.dot(H_R_KR_H, Phi_R)  # 维度：1*N

            h_k_R = np.dot(h_mid, self.H_B_R)  # 维度：1*M
            Signal_power_k_R = abs(np.dot(h_k_R, self.G[:, k_R])) ** 2  # calculate signal power  维度：1*M * M*1
            Interference_power_k_R = 0 - abs(np.dot(h_k_R, self.G[:, k_R])) ** 2  # calculate interference power
            for j_R in range(self.K):
                Interference_power_k_R += abs(np.dot(h_k_R, self.G[:, j_R])) ** 2

            SINR_k_R_numerator = Signal_power_k_R
            SINR_k_R_denominator_1 = Interference_power_k_R * (1 + self.k_B) + self.awgn_var * (1 + self.k_B)
            SINR_k_R_denominator_2 = Signal_power_k_R * self.k_B + h_k_R @ diag_tilde @ h_k_R.conj().T * (1 + self.k_B) * self.k_S
            SINR_k_R_denominator = SINR_k_R_denominator_1 + SINR_k_R_denominator_2

            SINR_k_R = SINR_k_R_numerator / SINR_k_R_denominator

            data_rate_k_R = math.log((1 + SINR_k_R), 2)  # calculate data rate
            self.data_rate_list_R[k_R] = data_rate_k_R  # saving data rate

        # calculate data rate for T users following the same way for R users
        for k_T in range(self.KT):
            H_R_KT_H = self.H_R_KT[:, k_T].conj().T
            h_mid = np.dot(H_R_KT_H, Phi_T)

            h_k_T = np.dot(h_mid, self.H_B_R)
            Signal_power_k_T = abs(np.dot(h_k_T, self.G[:, self.KR + k_T])) ** 2
            Interference_power_k_T = 0 - abs(np.dot(h_k_T, self.G[:, k_T])) ** 2
            for j_T in range(self.K):
                Interference_power_k_T += abs(np.dot(h_k_T, self.G[:, j_T])) ** 2

            SINR_k_T_numerator = Signal_power_k_T
            SINR_k_T_denominator_1 = Interference_power_k_T * (1 + self.k_B) + self.awgn_var * (1 + self.k_B)
            SINR_k_T_denominator_2 = Signal_power_k_T * self.k_B + h_k_T @ diag_tilde @ h_k_T.conj().T * (1 + self.k_B) * self.k_S
            SINR_k_T_denominator = SINR_k_T_denominator_1 + SINR_k_T_denominator_2

            SINR_k_T = SINR_k_T_numerator / SINR_k_T_denominator

            data_rate_k_T = math.log((1 + SINR_k_T), 2)
            self.data_rate_list_T[k_T] = data_rate_k_T

        reward = sum(self.data_rate_list_R) + sum(self.data_rate_list_T)

        return reward, opt_reward


    def step(self, action):
        self.episode_t += 1  # 总time step数？

        action = action.reshape(1, -1)  # 确保输入的动作是个行向量
        # 取出并还原G和Phi_R、Phi_T
        G_real = action[:, :self.M ** 2]
        G_imag = action[:, self.M ** 2:2 * self.M ** 2]

        self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)

        Phi_R_real = action[:, -4 * self.N:-3 * self.N]
        Phi_R_imag = action[:, -3 * self.N:-2 * self.N]

        Phi_T_real = action[:, -2 * self.N:-self.N]
        Phi_T_imag = action[:, -self.N:]

        self.Phi_R = np.eye(self.N, dtype=complex) * (Phi_R_real + 1j * Phi_R_imag)
        self.Phi_T = np.eye(self.N, dtype=complex) * (Phi_T_real + 1j * Phi_T_imag)

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2  # k维的行向量

        H_R_K_tilde_1, H_R_K_tilde_2 = self._compute_H_R_K_tilde()
        H_R_K_array = np.concatenate((H_R_K_tilde_1, H_R_K_tilde_2), axis=0)

        power_r = np.linalg.norm(H_R_K_array, axis=0).reshape(1, -1) ** 2  # k维的行向量

        H_R_KR_real, H_R_KR_imag = np.real(self.H_R_KR).reshape(1, -1), np.imag(self.H_R_KR).reshape(1, -1)
        H_R_KT_real, H_R_KT_imag = np.real(self.H_R_KT).reshape(1, -1), np.imag(self.H_R_KT).reshape(1, -1)
        H_B_R_real, H_B_R_imag = np.real(self.H_B_R).reshape(1, -1), np.imag(self.H_B_R).reshape(1, -1)

        self.state = np.hstack((action, power_t, power_r, H_R_KR_real, H_R_KR_imag, H_R_KT_real, H_R_KT_imag, H_B_R_real, H_B_R_imag))

        reward, opt_reward = self._compute_reward(self.Phi_R, self.Phi_T)

        done = opt_reward == reward

        return self.state, reward, done, None

    def close(self):
        pass
