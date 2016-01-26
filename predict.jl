using Mocha
backend = CPUBackend()
init(backend)

#input from memory rather than file
mem_data = MemoryDataLayer(name="data", tops=[:data], batch_size=1,
                           data=Array[zeros(Float32, 28, 28, 1, 1)])
#SoftmaxLayer instead of SoftmaxLossLayer because no backprop to be done
softmax_layer = SoftmaxLayer(name="prob", tops=[:prob], bottoms=[:ip2])

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

common_layers = [conv_layer, pool_layer, conv2_layer, pool2_layer,
                 fc1_layer, fc2_layer]

run_net = Net("imagenet", backend, [mem_data, common_layers..., softmax_layer])

#load last snapshot (10,000th iteration) generated from coffee break in lenet_train.jl
load_snapshot(run_net, "snapshots/snapshot-010000.jld")

#return class given 20x20 array
function predict(sample,net)
    get_layer(net,"data").data[1][:,:,1,1] = sample.data[:,:]
    forward(net)
    findmax(net.output_blobs[:prob].data)
end
function classIdToString(id)
    return string(labelMap[id])
end

for i in [1:size(xTest,1)]
    (prob,y) = predict(xTest[i],run_net)
    labelsInfoTest[i,:Class] = classIdToString(y)
end

writetable("$(path)/juliaSubmission.csv", labelsInfoTest, separator=',', header=true)

    


