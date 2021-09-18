

import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()
#import tensorflow as tf

import numpy as np
import time
from tensorflow import keras
from tensorflow.keras import layers
from itertools import product
np.set_printoptions(precision=4)

from tensorflow.keras import backend as Kend




def cal_RATE_NP(channel, tx_power, noise):
    user_num = channel.shape[1]

    diag_ch = np.zeros(channel.shape)
    for i in range(user_num):
        diag_ch[i, i] = channel[i, i]
    inter_ch = channel - diag_ch

    tot_ch = np.multiply(channel, tx_power)

    int_ch_1 = np.multiply(inter_ch, tx_power)
    sig_ch = np.sum(tot_ch - int_ch_1, axis=1)
    int_ch = np.sum(int_ch_1, axis=0)

    SINR_val = np.divide(sig_ch, int_ch + noise + 1e-10)
    cap_val = np.log(1.0 + SINR_val) * 1.4427

    return cap_val





def cal_RATE_EH_H_NP(channel_H, channel_G, tx_power, noise, harvesting_ratio):
    cap_val_DL = cal_RATE_NP(channel_H, tx_power, noise)
    EH_ch = np.dot(np.transpose(channel_G), tx_power)
    UL_power = harvesting_ratio*np.sum(EH_ch)
    return cap_val_DL, UL_power




def cal_RATE_G_NP(channel_G, tx_power, noise):
    Num_user = channel_G.shape[0]

    sig_ch = channel_G[0, 0] * tx_power[0, 0]
    int_ch = 0

    for i in range(Num_user - 1):
        int_ch = int_ch + channel_G[i + 1, 0] * tx_power[i + 1, 0]

    SINR_val = np.divide(sig_ch, int_ch + noise + 1e-10)
    cap_val = np.log(1.0 + SINR_val) * 1.4427

    return cap_val




def cal_TOTAL_NP(channel, tx_power, noise, tx_max, H_rat, R_th_tf, Eth_tf):
    CH_H = channel[:, :-1]
    CH_G = channel[:, -1:]

    PL_output = np.minimum(tx_power, 1.0) * tx_max
    H_rate, UL_power = cal_RATE_EH_H_NP(CH_H, CH_G, PL_output, noise, H_rat)
    G_rate = cal_RATE_G_NP(CH_G, PL_output, noise)
    R_out = np.float(np.any(R_th_tf > (H_rate[1:])))
    E_out = np.float(Eth_tf > UL_power)
    sec_rate = np.maximum(H_rate[0] - G_rate, 0) * (1 - R_out) * (1 - E_out)

    tot_out = np.float(sec_rate == 0)

    return sec_rate, UL_power, R_out, E_out, tot_out






def test_NP(channel, log_channel, model, sigma, tx_max, eta, Rth, Eth):
    Num_sample = channel.shape[0]
    Num_user = channel.shape[1]
    H_rate = []
    E_amount = []
    R_out = []
    E_out = []
    TOT_out = []
    s_time = time.time()
    tx_power = model.predict(log_channel)
    e_time = time.time()

    for i in range(Num_sample):
        H_rate_i, E_amount_i, R_out_i, E_out_i, Tot_out_i = cal_TOTAL_NP(channel[i], tx_power[i], sigma, tx_max, eta, Rth, Eth)
        H_rate.append(H_rate_i)
        E_amount.append(E_amount_i)
        R_out.append(R_out_i)
        E_out.append(E_out_i)
        TOT_out.append(Tot_out_i)

    print("")
    print("--------------- Averaging out of %d test channels ------------------" % (Num_sample))
    print("Average Sum Rate (bps/Hz) = ", np.mean(np.array(H_rate)))
    print("Rs Outage Probability (%) = ", 100 * np.mean(np.array(R_out)))
    print("EH Outage Probability (%) = ", 100 * np.mean(np.array(E_out)))
    print("TOT Outage Probability (%) = ", 100 * np.mean(np.array(TOT_out)))
    #print("Average EHs (mW) = ", np.mean(np.array(E_amount)))
    #print("Average Powers (mW) = ", tx_max * np.mean(np.array(np.reshape(tx_power, (-1, Num_user))), axis=0))
    print("--------------- END OF RESULT ------------------")
    print("")
    return np.mean(np.array(H_rate)), np.array(E_amount), 100*np.mean(np.array(R_out)), 100*np.mean(np.array(E_out)), 100*np.mean(np.array(TOT_out)), np.array(tx_power)





#######################################################################################################################################
#######################################################################################################################################
#########################                           TF CODE                                  ##########################################
#######################################################################################################################################
#######################################################################################################################################



def cal_RATE_tf(channel, PL_output, noise):
    #PL_rev = tf.expand_dims(PL_output, -1)
    PL_rev = PL_output
    tot_ch = tf.multiply(channel, PL_rev)
    sig_ch = tf.linalg.diag_part(tot_ch)
    inter_ch = tot_ch - tf.linalg.diag(sig_ch)
    inter_ch = tf.reduce_sum(inter_ch, axis=1)
    SINR_val = tf.divide(sig_ch, inter_ch + noise)
    cap_val = tf.math.log(tf.constant(1.0) + SINR_val) * tf.constant(1.4427)

    return cap_val





def cal_RATE_G_tf(channel, PL_output, noise):
    #PL_rev = tf.expand_dims(PL_output, -1)
    PL_rev = PL_output
    tot_ch = tf.multiply(channel, PL_rev)
    sig_ch = tot_ch[:,0,0]
    inter_ch = tf.reduce_sum(tot_ch[:,1:], axis=1)
    inter_ch = tf.reduce_sum(inter_ch, axis=1)
    SINR_val = tf.divide(sig_ch, inter_ch + noise)
    cap_val = tf.math.log(tf.constant(1.0) + SINR_val) * tf.constant(1.4427)

    return cap_val








def cal_RATE_EH_H_tf(channel, PL_output, noise, std_ch, harvesting_ratio, error_val):

    channel_init = channel * std_ch
    Num_D2D = int(channel_init.shape[2])


    multi_fading_I_err = channel_init[:,0,:,:] * ( (1 - error_val ** 2) ** 0.5 + error_val * tf.random.normal([Num_D2D, Num_D2D+1] ))
    multi_fading_Q_err = channel_init[:,1,:,:] * ( (1 - error_val ** 2) ** 0.5 + error_val * tf.random.normal([Num_D2D, Num_D2D+1] ))


    channel_tot = 0.5 * multi_fading_I_err ** 2 + 0.5 * multi_fading_Q_err ** 2


    channel_G = channel_tot[:, :,-1:]


    cap_val_DL = cal_RATE_tf(channel_H, PL_output, noise)
    cap_val_G = cal_RATE_G_tf(channel_G, PL_output, noise)

    PL_rev = PL_output
    EH_ch = tf.multiply(channel_G, PL_rev)
    UL_power = tf.reduce_sum(harvesting_ratio * EH_ch, axis=1)
    return cap_val_DL, UL_power, cap_val_G









def cal_LOSS_Total_tf(channel, tf_output, noise, tx_max, std_ch, H_rat, R_th_tf, Eth_tf, error_val):

    PL_output = tf_output
    PL_output = tf.minimum(PL_output, 1.0) * tx_max
    H_rate, UL_power, G_rate = cal_RATE_EH_H_tf(channel, PL_output, noise, std_ch, H_rat, error_val)
    #G_rate = cal_RATE_G_tf(CH_G, PL_output, noise, G_mean, G_mean)


    #Loss = - tf.constant(1.0) * tf.reduce_mean(H_rate)

    #Loss_2 = tf.reduce_mean(tf.constant(1.0) - tf.nn.relu(R_th_tf - (H_rate[:, 0] - G_rate)) / R_th_tf)

    Loss_2 = tf.reduce_mean(tf.constant(1.0) - tf.nn.relu((R_th_tf - H_rate[:, 1:]) / (R_th_tf + 1e-10)))

    #Loss_2 = tf.reduce_mean(tf.nn.relu(R_th_tf - (H_rate[:, 0])) / R_th_tf)
    Loss_3 = tf.reduce_mean(tf.constant(1.0) - tf.nn.relu((Eth_tf - UL_power) / Eth_tf))


    Loss = - tf.constant(1.0) * tf.reduce_mean(tf.nn.relu(H_rate[:,0] - G_rate)) * tf.pow(Loss_2, tf.constant(10.0)) * tf.pow(Loss_3,
                                                                                                   tf.constant(10.0))

    #Loss = tf.reduce_mean(tf.nn.relu((Eth_tf - UL_power) / Eth_tf))
    #Loss = - tf.reduce_mean(H_rate) + tf.constant(50.0) * Loss_3
    #Loss = -tf.pow(Loss_3, tf.constant(2.0))
    # Loss = tf.reduce_mean(tf.math.square(tf_output), axis=1)
    return Loss






def cal_LOSS_RATE_tf(channel, tf_output, noise, tx_max, std_ch, H_rat, R_th_tf, Eth_tf, error_val):

    PL_output = tf_output
    PL_output = tf.minimum(PL_output, 1.0) * tx_max
    H_rate, UL_power, G_rate = cal_RATE_EH_H_tf(channel, PL_output, noise, std_ch, H_rat, error_val)
    Loss = (tf.reduce_mean(tf.nn.relu(H_rate[:,0] - G_rate)))
    return Loss



def cal_LOSS_SECRECY_tf(channel, tf_output, noise, tx_max, std_ch, H_rat, R_th_tf, Eth_tf, error_val):

    PL_output = tf_output
    PL_output = tf.minimum(PL_output, 1.0) * tx_max
    H_rate, UL_power, G_rate = cal_RATE_EH_H_tf(channel, PL_output, noise, std_ch, H_rat, error_val)

    Loss = tf.reduce_mean(tf.math.ceil(tf.nn.relu(R_th_tf - H_rate[:,1:]) / (1e5)))

    return Loss






def cal_LOSS_EH_tf(channel, tf_output, noise, tx_max, std_ch, H_rat, R_th_tf, Eth_tf, error_val):
    PL_output = tf_output
    PL_output = tf.minimum(PL_output, 1.0) * tx_max
    H_rate, UL_power, G_rate = cal_RATE_EH_H_tf(channel, PL_output, noise, std_ch, H_rat, error_val)
    Loss = tf.math.ceil(tf.nn.relu(Eth_tf - UL_power)/10**5)
    return Loss








def Total_LOSS_wrapper(input_tensor, noise, tx_max, std_ch, H_rat, R_th_tf, Eth_tf, error_val):
    def TOTAL_LOSS_tf(y_true, y_pred):
        Loss = cal_LOSS_Total_tf(input_tensor, y_pred, noise, tx_max, std_ch, H_rat, R_th_tf, Eth_tf, error_val)
        return Loss
    return TOTAL_LOSS_tf




def RATE_LOSS_wrapper(input_tensor, noise, tx_max, std_ch, H_rat, R_th_tf, Eth_tf, error_val):
    def RATE_LOSS_tf(y_true, y_pred):
        Loss = cal_LOSS_RATE_tf(input_tensor, y_pred, noise, tx_max, std_ch, H_rat, R_th_tf, Eth_tf, error_val)
        return Loss
    return RATE_LOSS_tf


def SECRECY_LOSS_wrapper(input_tensor, noise, tx_max, std_ch, H_rat, R_th_tf, Eth_tf, error_val):
    def SECRECY_LOSS_tf(y_true, y_pred):
        Loss = cal_LOSS_SECRECY_tf(input_tensor, y_pred, noise, tx_max, std_ch, H_rat, R_th_tf, Eth_tf, error_val)
        return Loss
    return SECRECY_LOSS_tf

def EH_LOSS_wrapper(input_tensor, noise, tx_max, std_ch, H_rat, R_th_tf, Eth_tf, error_val):
    def EH_LOSS_tf(y_true, y_pred):
        Loss = cal_LOSS_EH_tf(input_tensor, y_pred, noise, tx_max, std_ch, H_rat, R_th_tf, Eth_tf, error_val)
        return Loss
    return EH_LOSS_tf


'''

    DNN MODELS

'''

def DNN_basic_module(Input_layer, Num_weights_inner, Num_outputs, Num_layers=3, activation='relu'):
    Inner_layer = layers.Dense(Num_weights_inner)(Input_layer)
    Inner_layer = layers.BatchNormalization()(Inner_layer)
    Inner_layer_2 = layers.Activation(activation)(Inner_layer)
    Inner_layer_in = Inner_layer_2

    ## Number of layers should be at least 2
    assert Num_layers > 1

    for i in range(Num_layers - 2):
        Inner_layer_in = layers.Dense(Num_weights_inner)(Inner_layer_in)
        Inner_layer_in = layers.BatchNormalization()(Inner_layer_in)
        Inner_layer_in = layers.Add()([Inner_layer_in, Inner_layer_2])
        Inner_layer_in = layers.Activation(activation)(Inner_layer_in)
        #Inner_layer_in = layers.Dropout(0.9)(Inner_layer_in)
        #Inner_layer = layers.Dropout(0.6)(Inner_layer)

    Out_layer = layers.Dense(Num_outputs)(Inner_layer_in)
    return Out_layer


def DNN_model(Num_user, Num_weights, Num_layers=4):
    inputs = tf.keras.Input(shape=(2, Num_user, Num_user+1))
    inputs_reshape = layers.Flatten(input_shape=(2, Num_user, Num_user+1))(inputs)

    ## Find the results for Power level
    result_PL = DNN_basic_module(inputs_reshape, Num_weights, Num_user, Num_layers)
    result_PL_2 = layers.Reshape((Num_user, 1))(result_PL)
    result_PL_3 = layers.Activation('sigmoid')(result_PL_2)

    model = tf.keras.Model(inputs=inputs, outputs=result_PL_3)

    return model






def gen_channel(Num_D2D, G_ref, Num_samples, error_var):
    PL_alpha = 36
    PL_const = 0

    H_distances = 20 + 30 * np.random.rand(Num_samples, Num_D2D, Num_D2D)
    direct = 5 + 10 * np.random.rand(Num_samples, Num_D2D)

    for k in range(Num_D2D):
        H_distances[:, k, k] = direct[:, k]


    ch_H_I = []
    ch_H_Q = []
    ch_H_I_err = []
    ch_H_Q_err = []


    for i in range(Num_samples):
        ## generate distance_vector
        dist_vec = H_distances[i]

        # find path loss // shadowing is not considered
        pu_ch_gain_db = - PL_const - PL_alpha * np.log10(dist_vec)
        pu_ch_gain = 10 ** (pu_ch_gain_db / 10)


        #multi_fading = 0.5 * np.random.randn(Num_D2D, Num_D2D) ** 2 + 0.5 * np.random.randn(Num_D2D, Num_D2D) ** 2
        H_ch_I = np.sqrt(pu_ch_gain) * np.random.randn(Num_D2D, Num_D2D)
        H_ch_Q = np.sqrt(pu_ch_gain) * np.random.randn(Num_D2D, Num_D2D)

        H_ch_I_err = H_ch_I * ( (1-error_var**2)**0.5 + error_var * np.random.randn(Num_D2D, Num_D2D) )
        H_ch_Q_err = H_ch_Q * ( (1-error_var**2)**0.5 + error_var * np.random.randn(Num_D2D, Num_D2D) )



        #multi_fading = 0.5 * multi_fading_I ** 2 + 0.5 * multi_fading_Q ** 2
        #multi_fading_err = 0.5 * multi_fading_I_err ** 2 + 0.5 * multi_fading_Q_err ** 2


        ch_H_I.append(H_ch_I)
        ch_H_Q.append(H_ch_Q)

        ch_H_I_err.append(H_ch_I_err)
        ch_H_Q_err.append(H_ch_Q_err)



    ch_H_I = np.array(ch_H_I)
    ch_H_Q = np.array(ch_H_Q)
    ch_H_I_err = np.array(ch_H_I_err)
    ch_H_Q_err = np.array(ch_H_Q_err)




    ch_G_I = []
    ch_G_Q = []
    ch_G_I_err = []
    ch_G_Q_err = []

    K_fact = 6  # K-factor in Rician fading

    G_distances = 5 + (10.0 - 5) * np.random.rand(Num_samples, Num_D2D)
    G_distances[:, 0] = 5



    for i in range(Num_samples):
        ## generate distance_vector
        dist_vec = G_distances[i]

        # find path loss // shadowing is not considered
        pu_ch_gain_db = - PL_const - PL_alpha * np.log10(dist_vec)
        pu_ch_gain = 10 ** (pu_ch_gain_db / 10)


        G_ch_I = np.sqrt(pu_ch_gain / 2 / (K_fact + 1)) * np.random.randn(Num_D2D)
        G_ch_Q = np.sqrt(pu_ch_gain * K_fact / (K_fact + 1)) + np.sqrt(
            pu_ch_gain / 2 / (K_fact + 1)) * np.random.randn(Num_D2D)



        G_ch_I_err = G_ch_I * ( (1-error_var**2)**0.5 + error_var * np.random.randn(Num_D2D) )
        G_ch_Q_err = G_ch_Q * ( (1-error_var**2)**0.5 + error_var * np.random.randn(Num_D2D) )

        data_g_test = 0.5 * G_ch_I ** 2 + 0.5 * G_ch_Q ** 2
        max_idx = np.argmax(data_g_test, axis=0)



        if max_idx != 0:
            G_ch_I[[0, max_idx]] = G_ch_I[[max_idx, 0]]
            G_ch_Q[[0, max_idx]] = G_ch_Q[[max_idx, 0]]


        ch_G_I.append(G_ch_I)
        ch_G_Q.append(G_ch_Q)

        ch_G_I_err.append(G_ch_I_err)
        ch_G_Q_err.append(G_ch_Q_err)


    ch_G_I = np.array(ch_G_I)
    ch_G_Q = np.array(ch_G_Q)
    ch_G_I_err = np.array(ch_G_I_err)
    ch_G_Q_err = np.array(ch_G_Q_err)



    return ch_H_I, ch_H_Q, ch_H_I_err, ch_H_Q_err, ch_G_I, ch_G_Q, ch_G_I_err, ch_G_Q_err







def prepro_chan(Num_user, G_ref, num_samples_tr, num_samples_val, error_var):

    ch_H_I, ch_H_Q, ch_H_I_err, ch_H_Q_err, ch_G_I, ch_G_Q, ch_G_I_err, ch_G_Q_err = gen_channel(Num_user, G_ref, num_samples_tr+num_samples_val, error_var)


    ## Finding the standard deivation of channel
    std_a = np.reshape(ch_H_I, [-1, ])
    std_b = np.reshape(ch_H_Q, [-1, ])
    std_c = np.reshape(ch_G_I, [-1, ])
    std_d = np.reshape(ch_G_Q, [-1, ])

    std_ch = np.concatenate([std_a, std_b, std_c, std_d]).std()



    ch_H_I_tr = ch_H_I[:num_samples_tr]
    ch_H_Q_tr = ch_H_Q[:num_samples_tr]
    ch_H_I_err_tr = ch_H_I_err[:num_samples_tr]
    ch_H_Q_err_tr = ch_H_Q_err[:num_samples_tr]

    ch_H_I_val = ch_H_I[num_samples_tr:]
    ch_H_Q_val = ch_H_Q[num_samples_tr:]
    ch_H_I_err_val = ch_H_I_err[num_samples_tr:]
    ch_H_Q_err_val = ch_H_Q_err[num_samples_tr:]


    ch_G_I_tr = ch_G_I[:num_samples_tr]
    ch_G_Q_tr = ch_G_Q[:num_samples_tr]
    ch_G_I_err_tr = ch_G_I_err[:num_samples_tr]
    ch_G_Q_err_tr = ch_G_Q_err[:num_samples_tr]

    ch_G_I_val = ch_G_I[num_samples_tr:]
    ch_G_Q_val = ch_G_Q[num_samples_tr:]
    ch_G_I_err_val = ch_G_I_err[num_samples_tr:]
    ch_G_Q_err_val = ch_G_Q_err[num_samples_tr:]




    ch_H_I_err_tr = ch_H_I_err_tr / std_ch
    ch_H_Q_err_tr = ch_H_Q_err_tr / std_ch
    ch_G_I_err_tr = ch_G_I_err_tr / std_ch
    ch_G_Q_err_tr = ch_G_Q_err_tr / std_ch


    ch_G_I_err_tr = np.expand_dims(ch_G_I_err_tr, axis=2)
    ch_G_Q_err_tr = np.expand_dims(ch_G_Q_err_tr, axis=2)


    ch_I_tot = np.concatenate((ch_H_I_err_tr, ch_G_I_err_tr), axis=2)
    ch_Q_tot = np.concatenate((ch_H_Q_err_tr, ch_G_Q_err_tr), axis=2)


    prep_data_tr = np.array([ch_I_tot, ch_Q_tot])
    prep_data_tr = np.transpose(prep_data_tr, [1, 0, 2, 3])








    ch_H_I_err_val = ch_H_I_err_val / std_ch
    ch_H_Q_err_val = ch_H_Q_err_val / std_ch
    ch_G_I_err_val = ch_G_I_err_val / std_ch
    ch_G_Q_err_val = ch_G_Q_err_val / std_ch


    ch_G_I_err_val = np.expand_dims(ch_G_I_err_val, axis=2)
    ch_G_Q_err_val = np.expand_dims(ch_G_Q_err_val, axis=2)


    ch_I_tot_val = np.concatenate((ch_H_I_err_val, ch_G_I_err_val), axis=2)
    ch_Q_tot_val = np.concatenate((ch_H_Q_err_val, ch_G_Q_err_val), axis=2)



    prep_data_val = np.array([ch_I_tot_val, ch_Q_tot_val])
    prep_data_val = np.transpose(prep_data_val, [1, 0, 2, 3])


    ch_H_val = 0.5 * ch_H_I_val ** 2 + 0.5 * ch_H_Q_val ** 2
    ch_G_val = 0.5 * ch_G_I_val ** 2 + 0.5 * ch_G_Q_val ** 2


    ch_G_val = np.expand_dims(ch_G_val, axis=2)

    data_test_raw = np.concatenate((ch_H_val, ch_G_val), axis=2)


    return prep_data_tr, prep_data_val, data_test_raw, std_ch










num_samples = int(1e6)
num_test_samples = int(1e5)


Num_weights = 200
Num_layers = 8

sigma = 10**(-10.0)

############
############
## PARAM to be CHANGED
############
############

G_ref = 5.0




############
############
############



Num_ch = 20000          ## Number of used samples for pretrain
Num_test = 1000

## learning_rate_PT is the learning rate for pre train
learning_rate_cur = 1e-5

validation_train_num = 10000




RATE_MAT_TOT = []
SECRECY_VIO_MAT_TOT = []
EH_VIO_MAT_TOT = []
TOT_VIO_MAT_TOT = []

for i in range(5):
    RATE_MAT_cur = []
    SECRECY_VIO_MAT_cur = []
    EH_VIO_MAT_cur = []


    batch_size_set = 1024 * 4
    num_iter_set = 500



    ################################################################
    ################################################################
    ################################################################
    ################################################################
    error_val = 0.05

    Rth_tf = 2.0

    Eth_dB = -15
    Eth = 10 ** (Eth_dB / 10.0)


    Pmax_dB = 23.0
    Pmax = 10 ** (Pmax_dB / 10.0)

    eta = 0.1 + 0.2*i

    K = 2
    ################################################################
    ################################################################
    ################################################################
    ################################################################







    print("%d-th iteration" % i)
    print("error_val = %f" % error_val)

    prep_data_tr, prep_data_val, data_test_raw, std_ch = prepro_chan(K, G_ref, num_samples, num_test_samples, error_val)



    #######################################################################################################
    #######################################################################################################
    ## DNN w/o pretrain
    #######################################################################################################
    #######################################################################################################
    Kend.clear_session()




    LABEL_temp = np.zeros((prep_data_tr.shape[0], 1))
    model_prop = DNN_model(K, Num_weights, Num_layers)

    model_prop.compile(optimizer=tf.train.AdamOptimizer(learning_rate_cur),
                       loss=Total_LOSS_wrapper(model_prop.input, sigma, Pmax, std_ch, eta, Rth_tf,
                                               Eth, error_val),
                       metrics=[
                           Total_LOSS_wrapper(model_prop.input, sigma, Pmax, std_ch, eta, Rth_tf,
                                              Eth, error_val),
                           RATE_LOSS_wrapper(model_prop.input, sigma, Pmax, std_ch, eta, Rth_tf,
                                             Eth, error_val),
                           SECRECY_LOSS_wrapper(model_prop.input, sigma, Pmax, std_ch, eta, Rth_tf,
                                                Eth, error_val),
                           EH_LOSS_wrapper(model_prop.input, sigma, Pmax, std_ch, eta, Rth_tf, Eth, error_val)
                       ])

    for iii in range(1):
        model_prop.fit(prep_data_tr, LABEL_temp, batch_size=batch_size_set, epochs=num_iter_set, verbose=2)

        print("")
        print("*" * 50)
        print("Proposed result with randomly generated test data")
        print("")
        RATE_MAT, EH_MAT, SECRECY_VIO_MAT, EH_VIO_MAT, TOT_VIO_MAT, TX_MAT = test_NP(data_test_raw, prep_data_val, model_prop, sigma,
                                                                        Pmax, eta, Rth_tf, Eth)


    print("")
    print("")
    print("")

    RATE_MAT_TOT.append(np.array(RATE_MAT))
    SECRECY_VIO_MAT_TOT.append(np.array(SECRECY_VIO_MAT))
    EH_VIO_MAT_TOT.append(np.array(EH_VIO_MAT))
    TOT_VIO_MAT_TOT.append(np.array(TOT_VIO_MAT))






print("--------------- Averaging out of test channels ------------------" )
print("Average Sum Rate (bps/Hz) = ", np.array(RATE_MAT_TOT))
print("Rs Outage Probability (%) = ", np.array(SECRECY_VIO_MAT_TOT))
print("EH Outage Probability (%) = ", np.array(EH_VIO_MAT_TOT))
print("TOT Outage Probability (%) = ", np.array(TOT_VIO_MAT_TOT))
print("--------------- END OF RESULT ------------------")
print("")