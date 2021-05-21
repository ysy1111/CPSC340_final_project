import os
import pickle
import argparse
import time
import numpy as np
from subprocess import run
import pandas as pd
from utils import euclidean_dist_squared as eds
import matplotlib.pyplot as plt

def Collect_Data(wdir):

    #collect data from 
    if wdir == "train":
        x_dir = train_dir_X
        y_dir = train_dir_y
    elif wdir == "test":
        x_dir = test_dir_X
    elif wdir == "val":
        x_dir = valid_dir_X
        y_dir = valid_dir_y
    
    os.chdir(x_dir)
    p_list = run('ls', capture_output=True, text=True,shell=True)
    file_list = p_list.stdout.split()
    pos_list_X = list()
    for i in range(len(file_list)):
        files = "./X_{0}.csv".format(i)
            #load the data set
        df = pd.read_csv(files)

        matrix = df.values
        index = np.where(matrix[0] == ' others')[0]
        index_agent = np.where(matrix[0] == ' agent')[0]
        pos_list_X.append((matrix[:,index+2],matrix[:,index+3],matrix[:,index_agent+2],matrix[:,index_agent+3])) 
    
    if wdir == "train" or wdir == "val":
        os.chdir(y_dir)
        p_list = run('ls', capture_output=True, text=True,shell=True)
        file_list = p_list.stdout.split()
        pos_list_y = list()
        for i in range(len(file_list)):
            files = "./y_{0}.csv".format(i)
                #load the data set
            df = pd.read_csv(files)

            pos_list_y.append((df.values[:,1],df.values[:,2]))

        return pos_list_X, pos_list_y
    else:
        return pos_list_X



def sum_distance_matrix(index, distance_matrix, min_car_number):
    idx, idy = zip(*index)
    total_distance = np.sort(distance_matrix[idx, idy])[0,:min_car_number]
    return np.sum(total_distance)/min_car_number

def image_distance(test_image,train_image,w,level = 1):
    #distance_matrix = np.matrix(np.ones((9,9)) * np.inf)
    pos_x_train = train_image[0]
    pos_y_train = train_image[1]

    pos_x_test = test_image[0]
    pos_y_test = test_image[1]
    res = 0
    if level == 1:
        real_dis_matrix = eds(pos_x_train.T,pos_x_test.T) + eds(pos_y_train.T,pos_y_test.T)
        #distance_matrix[:real_dis_matrix.shape[0],:real_dis_matrix.shape[1]] = real_dis_matrix[:]
        #print(distance_matrix)
        min_car_number = np.minimum(real_dis_matrix.shape[0],real_dis_matrix.shape[1])

        for i in range(min_car_number):
            res += np.min(real_dis_matrix)
            idx = np.where(np.min(real_dis_matrix))
            idy = np.where(np.min(real_dis_matrix))
            real_dis_matrix = np.delete(real_dis_matrix,idx,axis=0)
            real_dis_matrix = np.delete(real_dis_matrix,idy,axis=1)
        
        pos_agent_x_train = train_image[2] 
        pos_agent_y_train = train_image[3]

        pos_agent_x_test = test_image[2] 
        pos_agent_y_test = test_image[3]

        dis_between_agent_track = eds(pos_agent_x_train.T,pos_agent_x_test.T)+eds(pos_agent_y_train.T,pos_agent_y_test.T)
        return (1-w)*res/min_car_number + w*dis_between_agent_track[0,0]
    elif level == 0:
        pos_agent_x_train = train_image[2] 
        pos_agent_y_train = train_image[3]

        pos_agent_x_test = test_image[2] 
        pos_agent_y_test = test_image[3]

        dis_between_agent_track = eds(pos_agent_x_train.T,pos_agent_x_test.T)+eds(pos_agent_y_train.T,pos_agent_y_test.T)
    # print(dis_between_agent_track[0,0])
    # print(res/min_car_number)
        return dis_between_agent_track[0,0]
    

def flatten_coordinate(coordinates):
    print(coordinates.shape)
    x = coordinates[0]
    y = coordinates[1]
    xy = np.ones(2*len(x))
    xy[0::2] = x
    xy[1::2] = y
    return xy

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    
    #p_dir = run('cd  | pwd', capture_output=True, text=True,shell=True)
    #print(p_dir.stdout)
    main_dir = "/home/shuyy/CPSC340/final_part2/"
    train_dir_X = main_dir + "train/X"
    train_dir_y = main_dir + "train/y"
    valid_dir_X = main_dir + "val/X"
    valid_dir_y = main_dir + "val/y"
    test_dir_X = main_dir + "test/X"
    test_dir_y = main_dir + "test/y"

    


    if question == "KNN":
        print("KNN")


        train_X_list, train_y_list = Collect_Data("train")
        val_X_list, val_y_list = Collect_Data("val")
        #test_X_list, test_y_list = Collect_Data("test")
        #train_sample = [train_pos_list[-3]]

        w_set = np.arange(0.4,0,-0.2)
        k_set = np.arange(2,15,2)
        best_error = np.inf
        for w in w_set:
            for k in k_set:
                rmse = 0
                for car_number in range(len(val_X_list)):
                    #car_number = 240
                    valid_car = val_X_list[car_number]
                    result = list(map(lambda x: image_distance(valid_car, x,w=w), train_X_list)) 
                    #print(result)
                    index = np.argsort(result)[:k]
                    result = list(map(train_y_list.__getitem__, index))
                    #print(result)
                    good_result = list()
                    for i in result:
                        if len(i[0])==30:
                            good_result.append(i)
                    pred_y = np.mean(np.asarray(good_result), axis=0)
                    #print(pred_y)
                    #print(pred_y)
                    #print(val_y_list[0])
                    true_y = np.asarray(val_y_list[car_number])
                    #print(true_y-pred_y)
                    predx = pred_y[0]
                    predy = pred_y[1]
                    truex = true_y[0]
                    truey = true_y[1]
                    cur_error = np.sqrt(eds(predx.T,truex.T)+eds(predy.T,truey.T))
                    #print(car_number,index,cur_error)
                    rmse += cur_error[0,0]
                print("k: %d. w: %.2f .rmse: %.4f" % (k,w,rmse/len(val_X_list)))
                if rmse<best_error:
                    print("update the info")
                    best_error = rmse
                    best_k = k
                    best_w = w
        print("k: %d. w: %.2f .rmse: %.4f" % (best_k,best_w,best_error/len(val_X_list)))

    if question == "data_exploration":
        X_list, y_list = Collect_Data("train")
        #val_X_list, val_y_list = Collect_Data("val")
        for car_number in range(len(y_list)):
            y_pred = y_list[car_number]
            predy = y_pred[0]
            if len(predy)==29:
                print("29",car_number)
            elif len(predy)==28:
                print("28",car_number)
            elif len(predy)==27:
                print("27",car_number)

    if question == "only_agent":
        print("KNN")


        train_X_list, train_y_list = Collect_Data("train")
        val_X_list, val_y_list = Collect_Data("val")
        #test_X_list, test_y_list = Collect_Data("test")
        #train_sample = [train_pos_list[-3]]
        w = 1.0
        k_set = np.arange(1,15,2)
        #k_set=[4]
        best_error = np.inf
        for k in k_set:
            rmse = 0
            for car_number in range(len(val_X_list)):
                #car_number = 240
                valid_car = val_X_list[car_number]
                result = list(map(lambda x: image_distance(valid_car, x,w=w,level=0), train_X_list)) 
                #print(result)
                dist = np.sort(result)[:k]
                weight = (1/np.log(dist)) / (np.sum(1/np.log(dist))) 
                #print(weight)
                index = np.argsort(result)[:k]
                result = list(map(train_y_list.__getitem__, index))
                #print(result[0])
                good_result = list()
                for i,xy in enumerate(result):
                    x = xy[0] * weight[i]
                    y = xy[1] * weight[i]
                    good_result.append((x,y))
                pred_y = np.mean(np.asarray(good_result), axis=0)*k
                #print(pred_y)
                #print(pred_y)
                #print(val_y_list[0])
                true_y = np.asarray(val_y_list[car_number])
                #print(true_y-pred_y)
                predx = pred_y[0]
                predy = pred_y[1]
                truex = true_y[0]
                truey = true_y[1]
                cur_error = np.sqrt(eds(predx.T,truex.T)+eds(predy.T,truey.T))
                #print(car_number,index,cur_error)
                rmse += cur_error[0,0]
            print("k: %d.rmse: %.4f" % (k,rmse/len(val_X_list)))
            if rmse<best_error:
                print("update the info")
                best_error = rmse
                best_k = k
        print("k: %d.rmse: %.4f" % (best_k,best_error/len(val_X_list)))

    if question == "weighted_test_prediction":
        train_X_list, train_y_list = Collect_Data("train")
        #val_X_list, val_y_list = Collect_Data("val")
        test_X_list = Collect_Data("test")
        w = 1.0
        k = 6
        pred_y = []
        #print(len(test_X_list))
        for car_number in range(len(test_X_list)):
            test_car = test_X_list[car_number]
            result = list(map(lambda x: image_distance(test_car, x,w=w,level=0), train_X_list)) 
            dist = np.sort(result)[:k]
            weight = (1/dist) / (np.sum(1/dist)) 
            index = np.argsort(result)[:k]
            result = list(map(train_y_list.__getitem__, index))
            #print(weight)
            #print(result[0])
            good_result = list()
            for i,xy in enumerate(result):
                x = xy[0] * weight[i]
                y = xy[1] * weight[i]
                good_result.append((x,y))
            predy = np.mean(np.asarray(good_result), axis=0)*k
            pred_y = np.append(pred_y,flatten_coordinate(predy))
        
        os.chdir(test_dir_y)
        df = pd.read_csv("Shen_Ye_submission_2.csv")
        df["location"] = pred_y
        df.to_csv("Shen_Ye_submission_2.csv",index = False)
        #print(len(pred_y)/60)
    if question == "test_prediction":
        train_X_list, train_y_list = Collect_Data("train")
        #val_X_list, val_y_list = Collect_Data("val")
        test_X_list = Collect_Data("test")
        w = 1.0
        k = 10
        pred_y = []
        #print(len(test_X_list))
        for car_number in range(len(test_X_list)):
            test_car = test_X_list[car_number]
            result = list(map(lambda x: image_distance(test_car, x,w=w,level=0), train_X_list)) 
            index = np.argsort(result)[:k]
            result = list(map(train_y_list.__getitem__, index))

            pred_y = np.append(pred_y,flatten_coordinate(np.mean(np.asarray(result), axis=0)))
        
        os.chdir(test_dir_y)
        df = pd.read_csv("Shen_Ye_submission.csv")
        df["location"] = pred_y
        df.to_csv("Shen_Ye_submission.csv",index = False)
        #print(len(pred_y)/60)
    if question == "plot":
        df = pd.read_csv("../hyperplot.csv")

        scanw_k_2 = df[df["k"] == 2]
        scanw_k_4 = df[df["k"] == 4]
        scanw_k_6 = df[df["k"] == 6]
        scanw_k_8 = df[df["k"] == 8]
        scanw_k_10 = df[df["k"] == 10]
        scanw_k_12 = df[df["k"] == 12]
        scanw_k_14 = df[df["k"] == 14]

        scank_w_1 = df[df["w"] == 1] 
        x,y = scanw_k_2["w"], scanw_k_2["error"]    
        f1, = plt.plot(x,y, color='b', marker='o', linestyle='dashed',linewidth=2, markersize=12)

        x,y = scanw_k_4["w"], scanw_k_4["error"]
        f2, = plt.plot(x,y, color='g', marker='o', linestyle='dashed',linewidth=2, markersize=12)
        
        x,y = scanw_k_6["w"], scanw_k_6["error"]
        f3, = plt.plot(x,y, color='r', marker='o', linestyle='dashed',linewidth=2, markersize=12)

        x,y = scanw_k_8["w"], scanw_k_8["error"]
        f4, = plt.plot(x,y, color='c', marker='o', linestyle='dashed',linewidth=2, markersize=12)

        x,y = scanw_k_10["w"], scanw_k_10["error"]
        f5, = plt.plot(x,y, color='m', marker='o', linestyle='dashed',linewidth=2, markersize=12)

        x,y = scanw_k_12["w"], scanw_k_12["error"]
        f6, = plt.plot(x,y, color='y', marker='o', linestyle='dashed',linewidth=2, markersize=12)

        x,y = scanw_k_14["w"], scanw_k_14["error"]
        f7, = plt.plot(x,y, color='k', marker='o', linestyle='dashed',linewidth=2, markersize=12)

        plt.legend([f1, f2, f3, f4, f5, f6, f7], ['k=2', 'k=4', 'k=6','k=8', 'k=10', 'k=12','k=14'])
        plt.xlabel('w')
        plt.ylabel('Error')
        path = os.path.join('..', 'figs', "scan_w")
        plt.savefig(path)
        plt.close()

        x,y = scank_w_1["k"], scank_w_1["error"]    
        f1, = plt.plot(x,y, color='b', marker='o', linestyle='dashed',linewidth=2, markersize=12, label='w=1')
        plt.xlabel('k')
        plt.ylabel('Error')
        path = os.path.join('..', 'figs', "scan_k")
        plt.savefig(path)
