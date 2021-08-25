import sparsechem as sc
from collaborative.participant import Server, Client
import utils.data_utils as du
import random
import numpy as np
from tqdm import tqdm
import torch


def hide_h_percent(X_train, Y_train, h):
    X_train_num = X_train.shape[0]
    print("All data: ", X_train_num)
    remaining_data_num = int((1 - h) * X_train_num)
    shared_data_indices = random.sample(range(X_train_num), remaining_data_num)
    print("After hiding %d%%: %d" % (h * 100, remaining_data_num))
    return X_train[shared_data_indices], Y_train[shared_data_indices]


def run_server(group, hide, conf, rounds, server_data, client_data_path, same_head=False):
    trunk = sc.Trunk(conf)
    loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    ## Server
    ecfp_va, ic50_va = server_data
    dataset_va = sc.SparseDataset(ecfp_va, ic50_va)
    server = Server(trunk, conf=conf, dataset=dataset_va, loss=loss)

    if same_head:
        model = sc.TrunkAndHead(conf=conf, trunk=trunk)

    clients = []
    train_path = client_data_path + "data_2_split/"
    for k, h in zip(group, hide):
        # Load client train data
        X_train, Y_train = du.load_ratio_split_data(train_path, k, train=True)
        if h != 0:
            # Hide h% of the train data
            X_train, Y_train = hide_h_percent(X_train, Y_train, h)
        dataset = sc.SparseDataset(X_train, Y_train)
        # Load client test data
        X_data, Y_data = du.load_ratio_split_data(train_path, k, train=False)
        dataset_va = sc.SparseDataset(X_data, Y_data)
        ## Client
        conf.output_size = Y_train.shape[1]
        conf.batch_size = int(X_train.shape[0] * 0.02)
        print("Batch size: %d" % conf.batch_size)
        if not same_head:
            model = sc.TrunkAndHead(conf=conf, trunk=trunk)
        print("Trunk + Head architecture (client-%d):" % k)
        print(model)
        client = Client(model, conf=conf, dataset=dataset, dataset_va=dataset_va)
        clients.append(client)

    # Evaluate before training
    loss_o_arr = []
    for c in clients:
        loss, _ = c.eval(on_train=False)
        loss_o_arr.append(loss)
    loss_o_arr = np.array(loss_o_arr)

    ## run model
    for r in tqdm(range(rounds)):
        for i, client in enumerate(clients):
            # client gets his next batch
            batch = client.get_next_batch()
            # client calculates updates for trunk+head on that batch
            client.train(batch)
            # client updates his head
            client.update_weights()
            # client zeroes his head
            client.zero_grad()
        # server updates his trunk
        server.update_weights()
        # server zeroes his trunk
        server.zero_grad()

    # Evaluate after training
    loss_arr = []
    for c in clients:
        loss, _ = c.eval(on_train=False)
        loss_arr.append(loss)
    loss_arr = np.array(loss_arr)
    return loss_o_arr, loss_arr
