import numpy as np  
import matplotlib.pyplot as plt  
 
# 读取存储为txt文件的数据  
def data_read(dir_path):  
    with open(dir_path, "r") as f:  
        raw_data = f.read()  
        data = raw_data[1:-1].split(", ")  
    return np.asarray(data, dtype=float)  
  
  
#不同长度数据,统一为一个标准,倍乘x轴  
def multiple_equal(x, y):  
    x_len = len(x)  
    y_len = len(y)  
    times = x_len/y_len
    return [i * times for i in y]

if __name__ == "__main__":
    train_loss_path = "/workspace/project_flux/x-flux/logs/losses_20250102-171545.txt"
    train_loss_path = "/workspace/project_flux/x-flux/logs/losses_20250102-210519.txt"
    # train_acc_path = r"E:\relate_code\Gaitpart-master\train_acc.txt"
    y_train_loss = data_read(train_loss_path)
    # y_train_acc = data_read(train_acc_path)
    x_train_loss = range(len(y_train_loss))
    # x_train_acc = multiple_equal(x_train_loss, range(len(y_train_acc)))

    plt.figure()  
    ax = plt.axes()  
    ax.spines['top'].set_visible(False)  
    ax.spines['right'].set_visible(False)  

    plt.plot(x_train_loss, y_train_loss, label='Training Loss', color='blue')  
    plt.xlabel('Iterations')  
    plt.ylabel('Loss')  
    plt.title('Training Loss Over Iterations')  
    plt.legend()  

    # Save the plot as an image file  
    plt.savefig('../saves/training_loss_plot_40epoch.png', bbox_inches='tight')  
    #plt.show() 
