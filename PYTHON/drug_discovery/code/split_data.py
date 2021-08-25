import random
import numpy as np
import utils.data_utils as du


def split_with_overlap(ratio, ecfp_tr, ic50_tr, root_dir="", overlap=1000):
    print("Split with %d overlap." % overlap)
    split_path = root_dir + "data_2_split/"

    training_sample_num = ecfp_tr.shape[0]
    one_part = int(training_sample_num / (ratio + 1))
    samples_user_1 = ratio * one_part
    samples_user_2 = one_part
    print("Number of training samples user-1: %d" % (samples_user_1))
    print("Number of training samples user-2: %d" % (samples_user_2))

    shuffled_idx = np.array(range(training_sample_num))
    np.random.shuffle(shuffled_idx)
    user_1 = shuffled_idx[:samples_user_1]
    user_2 = shuffled_idx[-samples_user_2:]

    user_1_train_size = int(0.8 * samples_user_1)
    user_2_train_size = int(0.8 * samples_user_2)

    user_1_train = user_1[:user_1_train_size]
    user_1_test = user_1[user_1_train_size:]
    user_2_train = user_2[:user_2_train_size]
    user_2_test = user_2[user_2_train_size:]

    # T_total - number of targets
    T_total = ic50_tr.shape[1]
    # number of disjunct targets
    num_disjunct = (T_total - overlap) // 2

    print("%d disjunct and %d overlapping labels (total: %d)" % (num_disjunct, overlap, T_total))

    shuffled_labels = np.array(range(T_total))
    np.random.shuffle(shuffled_labels)
    common_labels = shuffled_labels[:overlap]
    if num_disjunct == 0:
        user_1_labels, user_2_labels = common_labels, common_labels
    else:
        user_1_labels = np.concatenate((common_labels, shuffled_labels[overlap:overlap + num_disjunct]), axis=None)
        user_2_labels = np.concatenate((common_labels, shuffled_labels[-num_disjunct:]), axis=None)

    u1_ic50_tr = ic50_tr.tocsc()[:, user_1_labels].tocsr()
    u2_ic50_tr = ic50_tr.tocsc()[:, user_2_labels].tocsr()

    ## save user-1 data
    du.save_data(split_path + "0_train/", ecfp_tr[user_1_train],
                 u1_ic50_tr[user_1_train])
    du.save_data(split_path + "0_test/", ecfp_tr[user_1_test],
                 u1_ic50_tr[user_1_test])

    ## save user-2 data
    du.save_data(split_path + "1_train/", ecfp_tr[user_2_train],
                 u2_ic50_tr[user_2_train])
    du.save_data(split_path + "1_test/", ecfp_tr[user_2_test],
                 u2_ic50_tr[user_2_test])
