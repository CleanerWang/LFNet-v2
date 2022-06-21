import numpy as np

test = open("../2007_test.txt","w")
with open("../2007_train.txt","r") as f:
    all_file = f.readlines()
    np.random.shuffle(all_file)
    k=0
    for file in all_file:
        if k<100:
            test.write(file)
        else:
            break
        k+=1

test.close()