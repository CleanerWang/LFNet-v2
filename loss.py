import glob
import matplotlib.pyplot as plt
files = glob.glob("../train-log/*.txt" )
all_line = []
for file in files:
    with open(file,"r",encoding="utf-8") as f:
        lline = f.readlines()
        for ll in lline:
            all_line.append(ll)
dict = {}
k=0
for line in all_line:
    if line.find("step - loss:")>0 and line.find("- val_loss:")>0 :
        value =line.split("step - loss:")[-1].split("- val_loss:")[0].strip()
        print(value)
        dict[k] = float(value)
        k+=1

test_data_1=sorted(dict.items(),key=lambda x:x[0])
print(test_data_1)
x_axis = []
y_axis =[]
for item in test_data_1:
    x_axis.append(item[0])
    y_axis.append(item[1])
plt.title('train loss')
plt.plot(x_axis[5:], y_axis[5:], color='green', label='training loss')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
