# SGLD

SGLD is implemented in the file sgld_optimizer.py

run_nn.py is the main script and takes arguments {net_type, dataset, optim, lr, batch_size, epochs, key}

Either use dataset "abalone" and net_type "MLP" or dataset "CIFAR10" and net_type "AlexNet"

Setting optim to "both" will train the model twice and compare the results

Plots are saved to the "results" subdirectory.

Model state information is stored in the "model_info" directory in a file corresponding to the key provided at runtime.  If a file exists corresponding to that key, the state will be recovered.  If no such file exists, a new model is trained and a file is created to store the state.
