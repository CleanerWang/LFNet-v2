classes = ["smoke"]
all_files = []
with open("../2007_train.txt","r") as f: # fig7_test.txt
    all_files = f.readlines()
    for file in all_files:
        filename= file.split(" ")[0].split("/")[-1].split(".")[0]
        with open("./groundtruths/"+filename+".txt","w") as ff:
            infos = file.split(" ")[1:]
            for info in infos:
                classname = classes[int(info.split(",")[-1].strip())]
                result = " ".join(info.split(",")[:-1])
                ff.write(classname + " " + result+"\n")

