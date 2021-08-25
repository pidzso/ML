import utils.data_utils as du
import numpy as np
import torch
import CollaborativeLearning as CoL
import sparsechem as sc
import csv, os
from split_data import split_with_overlap

client_num = 2
rounds = 200
num_tr_samples = 236182
## batch size should be 2% of the training samples of 1 client
batch_size = int((num_tr_samples/client_num)*0.02)

## model configuration
conf = sc.ModelConfig(
    input_size         = 32000,
    hidden_sizes       = [40],
    output_size        = 2808,
    batch_size         = batch_size,
    lr                 = 1e-3,
    last_dropout       = 0.2,
    weight_decay       = 1e-5,
    non_linearity      = "relu",
    last_non_linearity = "relu",
    optimizer="ADAM"
)

ecfp_tr, ic50_tr, ecfp_va, ic50_va, = du.load_data("../data/")
ecfp_tr, ecfp_va = du.fold_input(ecfp_tr, 32000), du.fold_input(ecfp_va, 32000)


self_division_data_path = "self_division_data/"
if not os.path.exists(self_division_data_path):
    print("Split data between 2 users")
    os.makedirs(self_division_data_path)
    du.make_split_save(2, ecfp_tr, ic50_tr, root_dir=self_division_data_path)


def reinitialise_seed(seed_init):
    np.random.seed(seed_init)
    torch.manual_seed(seed_init)


def run_server_wrapper(group, hide, data_path):
    return CoL.run_server(group=group, hide=hide, conf=conf, rounds=rounds, server_data=(ecfp_va, ic50_va),
                          client_data_path=data_path, same_head=True)


def both_hide_experiments(ecfp_tr, ic50_tr, data_path, overlap, results_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    ratios = [1]

    for seed_init in range(3):
        for ratio in ratios:
            split_with_overlap(ratio, ecfp_tr, ic50_tr, root_dir=data_path, overlap=overlap)

            # train alone
            reinitialise_seed(seed_init)
            o_1, theta_1 = run_server_wrapper([0], [0], data_path)
            reinitialise_seed(seed_init)
            o_2, theta_2 = run_server_wrapper([1], [0], data_path)
            o_1, theta_1 = o_1[0], theta_1[0]
            o_2, theta_2 = o_2[0], theta_2[0]

            hidings = [0, 0.2, 0.4, 0.6, 0.8]
            p1_h = len(hidings) * hidings
            p2_h = [h for h in hidings for _ in range(len(hidings))]

            for h1, h2 in zip(p1_h, p2_h):
                print("User-1 hides: %d%%, User-2 hides: %d%%" % (h1 * 100, h2 * 100))
                # train together
                reinitialise_seed(seed_init)
                _, Theta = run_server_wrapper([0, 1], [h1, h2], data_path)
                Theta_1, Theta_2 = Theta[0], Theta[1]
                print("Theta: ", Theta)

                # calculate normalised accuracy improvement
                norm_acc_impr_1 = ((o_1 - Theta_1) - (o_1 - theta_1)) / (o_1 - theta_1)
                print("(1) Accuracy improvement: ", norm_acc_impr_1)
                norm_acc_impr_2 = ((o_2 - Theta_2) - (o_2 - theta_2)) / (o_2 - theta_2)
                print("(2) Accuracy improvement: ", norm_acc_impr_2)

                results = {}
                results["ratio"] = str(ratio) + ":1"
                results["hidden"] = h1
                results["o"] = o_1
                results["theta"] = theta_1
                results["Theta"] = Theta_1
                results["accuracy_improvement"] = norm_acc_impr_1

                results_2 = {}
                results_2["ratio"] = "1:" + str(ratio)
                results_2["hidden"] = h2
                results_2["o"] = o_2
                results_2["theta"] = theta_2
                results_2["Theta"] = Theta_2
                results_2["accuracy_improvement"] = norm_acc_impr_2

                with open(results_path, "a+") as f:
                    writer = csv.DictWriter(f, list(results.keys()))
                    if os.path.getsize(results_path) == 0:
                        writer.writeheader()
                    writer.writerow(results)
                    writer.writerow(results_2)


if __name__ == "__main__":
    split_data_path = os.path.join(self_division_data_path, "data_2_split/")
    print("Load data from 1st user...")
    ecfp_u1, ic50_u1 = du.load_split_data(split_data_path, k=0)
    print("Run both hide experiment with the 1st user's data")
    both_hide_experiments(ecfp_u1, ic50_u1, data_path="data4_u1/", overlap=2808,
                          results_path="self_division_same_head_u1_both_hide.csv")

    print("Load data from 2nd user...")
    ecfp_u2, ic50_u2 = du.load_split_data(split_data_path, k=1)
    print("Run both hide experiment with the 2nd user's data")
    both_hide_experiments(ecfp_u2, ic50_u2, data_path="data4_u2/", overlap=2808,
                          results_path="self_division_same_head_u2_both_hide.csv")