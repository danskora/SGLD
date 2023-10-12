# SGLD

SGLD is implemented in the file sgld_optimizer.py

run_nn.py is the main script and takes args {net_type, dataset, optim, lr, batch_size, epochs}

Either use dataset "abalone" and net_type "MLP" or dataset "CIFAR10" and net_type "AlexNet"

Setting optim to "both" will train the model twice and compare the results

Plots are saved to the results subdirectory
