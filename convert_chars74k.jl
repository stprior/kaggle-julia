using Images
using DataFrames
using HDF5
#typeData could be either "train" or "test.
#labelsInfo should contain the IDs of each image to be read
#The images in the trainResized and testResized data files
#are 20x20 pixels, so imageSize is set to 400.
#path should be set to the location of the data files.

function read_data(typeData, labelsInfo, imageSize, path)
    #Intialize x matrix
    x = Array{Image{Colors.Gray{Float32}}}(size(labelsInfo,1))
    
    for (index, idImage) in enumerate(labelsInfo[:ID])
        #Read image file
        nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
        img = load(nameFile)

        #Convert img to float values
        #Convert color images to gray images
        img  = convert(Image{Colors.Gray{Float32}},img)
        
        x[index] = img
        
    end
    return x
end

imageSize = 400 # 20 x 20 pixel

#Set location of data files, folders
path = "/home/stprior/kaggle"

#Read information about training data , IDs.
labelsInfoTrain = readtable("$(path)/trainLabels.csv")

#Read training matrix
xTrain = read_data(:train, labelsInfoTrain, imageSize, path)

#Read information about test data ( IDs ).
labelsInfoTest = readtable("$(path)/sampleSubmission.csv")

#Read test matrix
xTest = read_data(:test, labelsInfoTest, imageSize, path)

#Get only first character of string (convert from string to character).
#Apply the function to each element of the column "Class"
yTrain = map(x -> x[1], labelsInfoTrain[:Class])

#Mocha expects labels between 0 and the number of classes (0:61) - this maps the char to array value
function stringToClassId(str)
    ch = str[1]
    labelMap = [collect('0':'9');'A':'Z';'a':'z']
    return find(e -> e==ch,labelMap)-1
end
    
#Convert from character to integer
yTrain = map(yTrain) do label
    label = stringToClassId(label)
end



function writeHd5File(name, images, labels)

    h5open("$(name).hdf5", "w") do h5
        h = height(images[1])
        w = width(images[1])
        n_data = size(labels,1)
        println("Exporting $n_data digits of size $h x $w")

        dset_data = d_create(h5, "data", datatype(Float32), dataspace(w, h, 1, n_data))
        dset_label = d_create(h5, "label", datatype(Float32), dataspace(1, n_data))

        for idx = 1:n_data
            dset_data[:,:,1,idx] = images[idx].data[:,:]
            dset_label[1,idx] = labels[idx]
        end
    end
end

writeHd5File("train",xTrain,yTrain)
#writeHd5File("test",xTest,yTrain)
    

