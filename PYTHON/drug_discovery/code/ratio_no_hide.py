import utils.data_utils as du
import numpy as np
import torch
import CollaborativeLearning as CoL
import sparsechem as sc
import csv, os
from split_data import split_with_overlap

data_path = "data/"
results_path = "ratio_no_hide_same_head.csv"

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
)

ecfp_tr, ic50_tr, ecfp_va, ic50_va, = du.load_data("../data/")
ecfp_tr, ecfp_va = du.fold_input(ecfp_tr, 32000), du.fold_input(ecfp_va, 32000)

def reinitialise_seed(seed_init):
    np.random.seed(seed_init)
    torch.manual_seed(seed_init)

def run_server_wrapper(group, hide):
    return CoL.run_server(group=group, hide=hide, conf=conf, rounds=rounds, server_data=(ecfp_va, ic50_va),
                          client_data_path=data_path, same_head=True)

if __name__ == "__main__":
    ratios = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    for seed_init in range(3):
        for ratio in ratios:
            split_with_overlap(ratio, ecfp_tr, ic50_tr, root_dir=data_path, overlap=2808)

            # train alone
            reinitialise_seed(seed_init)
            o_1, theta_1 = run_server_wrapper([0], [0])
            reinitialise_seed(seed_init)
            o_2, theta_2 = run_server_wrapper([1], [0])
            o_1, theta_1 = o_1[0], theta_1[0]
            o_2, theta_2 = o_2[0], theta_2[0]

            # train together
            reinitialise_seed(seed_init)
            _, Theta = run_server_wrapper([0,1], [0,0])
            Theta_1, Theta_2 = Theta[0], Theta[1]
            print("Theta: ", Theta)

            # calculate normalised accuracy improvement
            norm_acc_impr_1 = ((o_1 - Theta_1) - (o_1 - theta_1)) / (o_1 - theta_1)
            print("(1) Accuracy improvement: ", norm_acc_impr_1)
            norm_acc_impr_2 = ((o_2 - Theta_2) - (o_2 - theta_2)) / (o_2 - theta_2)
            print("(2) Accuracy improvement: ", norm_acc_impr_2)

            results = {}
            results["ratio"] = str(ratio) + ":1"
            results["o"] = o_1
            results["theta"] = theta_1
            results["Theta"] = Theta_1
            results["accuracy_improvement"] = norm_acc_impr_1

            results_2 = {}
            results_2["ratio"] = "1:" + str(ratio)
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