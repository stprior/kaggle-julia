# convert binary into HDF5 data

using HDF5
using Compat
using Images
using DataFrames
using Mocha

srand(12345678)

data_layer  = AsyncHDF5DataLayer(name="train-data", source="train.txt",
                            batch_size=64, shuffle=true)

conv_layer = ConvolutionLayer(name="conv1", n_filter=20, kernel=(5,5),
                              bottoms=[:data], tops=[:conv1])

pool_layer = PoolingLayer(name="pool1", kernel=(2,2), stride=(2,2),
                          bottoms=[:conv1], tops=[:pool1])

conv2_layer = ConvolutionLayer(name="conv2", n_filter=50, kernel=(5,5),
                               bottoms=[:pool1], tops=[:conv2])
pool2_layer = PoolingLayer(name="pool2", kernel=(2,2), stride=(2,2),
                           bottoms=[:conv2], tops=[:pool2])

fc1_layer  = InnerProductLayer(name="ip1", output_dim=500,
                               neuron=Neurons.ReLU(), bottoms=[:pool2], tops=[:ip1])
fc2_layer  = InnerProductLayer(name="ip2", output_dim=62,
                               bottoms=[:ip1], tops=[:ip2])

loss_layer = SoftmaxLossLayer(name="loss", bottoms=[:ip2,:label])

backend = CPUBackend()
init(backend)

common_layers = [conv_layer, pool_layer, conv2_layer, pool2_layer,
                 fc1_layer, fc2_layer]
net = Net("MNIST-train", backend, [data_layer, common_layers..., loss_layer])

exp_dir = "snapshots"
method = SGD()
params = make_solver_parameters(method, max_iter=10000, regu_coef=0.0005,
                                mom_policy=MomPolicy.Fixed(0.9),
                                lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
                                load_from=exp_dir)
solver = Solver(method, params)

setup_coffee_lounge(solver, save_into="$exp_dir/statistics.hdf5", every_n_iter=1000)

add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

#data_layer_test = HDF5DataLayer(name="test-data", source="data/test.txt", batch_size=100)
#acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:ip2, :label])
#test_net = Net("MNIST-test", backend, [data_layer_test, common_layers..., acc_layer])

#add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

solve(solver, net)

destroy(net)
destroy(test_net)
shutdown(backend)
