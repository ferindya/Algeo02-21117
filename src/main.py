from extractor import *
from eigenface import *

imgList, nameList = batch_extractor("src/train")
eigenfaces, mean, weightTrain = train(imgList)
imgTest = extractImg("src/test/Agna2.jpg",500)
closest_idx, dist = test(imgTest,eigenfaces,mean,weightTrain)
print(nameList[closest_idx],dist)