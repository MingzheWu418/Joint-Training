import os
import numpy as np
import sys
import torch.utils.data
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import pandas as pd
import json


File_names = ['co2.csv', 'humidity.csv', 'light.csv', 'temperature.csv']
# File_names = ['co2.csv', 'humidity.csv', 'temperature.csv']

def clean_coequipment(ts, val, maxl = 30000):
    new_ts = [ts[0]]
    new_val = [val[0]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] == 1500:
            new_val.append(val[i])
        else:
            k = int((ts[i] - ts[i - 1]) / 1500)
            for _ in range(k):
                new_val.append(val[i])
    return new_val[0:maxl]

def read_ahu_csv(path, column = ['PropertyTimestampInNumber', 'SupplyFanSpeedOutput']):
    df = pd.read_csv(path)
    ts = df[column[0]]
    val = df[column[1]]
    return clean_coequipment(ts, val)

def read_vav_csv(path, column = ['PropertyTimestampInNumber', 'AirFlowNormalized']):
    df = pd.read_csv(path)
    ts = df[column[0]]
    val = df[column[1]]
    return clean_coequipment(ts, val)

def read_facility_ahu(facility_id, ahu_list):
    ahu_data, label = [], []
    path = "/localtmp/sl6yu/split/ahu_property_file_" + str(facility_id) + "/"
    for name in ahu_list[facility_id]:
        if os.path.exists(path + name + '.csv') == False:
            continue
        label.append(name)
        ahu_data.append(read_ahu_csv(path + name + '.csv'))
    return ahu_data, label

def read_facility_vav(facility_id, mapping):
    vav_data, label = [], []
    path = "/localtmp/sl6yu/split/vav_box_property_file_" + str(facility_id) + "/"
    for name in mapping[facility_id].keys():
        if os.path.exists(path + name + '.csv') == False:
            continue
        label.append(name)
        vav_data.append(read_vav_csv(path +name + '.csv'))
    return vav_data, label

def read_ground_truth(building, path = './mapping_data.xlsx'):
    roomList = []
    if building == "Soda":
        f = open("./rawdata/groundtruth/SODA-GROUND-TRUTH", "r+")
        lines = f.readlines()
        i = 0
        while i < len(lines) - 1:
            sensorName = lines[i].strip()
            roomCorr = [sensorName]
            i += 1
            currLine = lines[i].strip()
            '''
            Manually consider all cases.
            If given more information, can rewrite in a more elegant way
            '''
            if currLine.find("room-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(str(currLine[3]) + ", " + str(currLine[4]))
                # we need both room name and room id here
                roomList.append(roomCorr)
            '''
            elif currLine.find("chilled/condensor water loop-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(currLine[2])
                roomList.append(roomCorr)
            elif currLine.find("supply fan-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(currLine[4])
                roomList.append(roomCorr)
            elif currLine.find("ahu-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(currLine[2])
                roomList.append(roomCorr)
            elif currLine.find("hot water loop-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(currLine[2])
                roomList.append(roomCorr)
            elif currLine.find("chiller-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(currLine[2])
                roomList.append(roomCorr)
            elif currLine.find("exhaust fan-id") != -1:
                currLine = currLine.split(",")
                try:
                    roomCorr.append(currLine[4])
                except IndexError:
                    roomCorr.append(currLine[2])
                roomList.append(roomCorr)
            elif currLine.find("condensor pump-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(currLine[2])
                roomList.append(roomCorr)
            else:
                pass
            '''

            i += 1

        f.close()
        # print(roomList)
        # [['soda...', 'room:123'], [], ... ]
        return roomList
    elif building == "SDH":
        f = open("./rawdata/groundtruth/sdh_tagsets.json", "r+")
        data = json.load(f)
        # print(data)
        # for i in data:
        #     print(i)
        #     print(data[i])
        return
    else:
        data = pd.read_excel(path, sheet_name = 'Hierarchy Data', usecols=[1, 6, 7, 9])
        raw_list = data.values.tolist()
        mapping = dict()
        ahu_vas = dict()
        ahu_list = dict()
        for line in raw_list:
            if line[3] != 'AHU':
                continue
            f_id = int(line[0])
            parent_name = line[1]
            child_name = line[2]
            if f_id not in ahu_vas.keys():
                ahu_vas[f_id] = dict()
                ahu_list[f_id] = []
            ahu_list[f_id].append(child_name)
            ahu_vas[f_id][parent_name] = child_name

        for line in raw_list:
            if line[3] != 'VAV-BOX':
                continue
            f_id = int(line[0])
            parent_name = line[1]
            child_name = line[2]
            if f_id not in mapping.keys():
                mapping[f_id] = dict()
            if parent_name in ahu_vas[f_id].keys():
                mapping[f_id][child_name] = ahu_vas[f_id][parent_name]
        return mapping, ahu_list

    
def clean_temperature(value):
    for i in range(len(value)):
        if value[i] > 40 or value[i] < 10:
            if i == 0:
                value[i] = value[i + 1]
            else:
                value[i] = value[i - 1]
    return value

def read_colocation(config):
    folders = os.walk(config.data)
    x = []
    y = []
    true_pos = []
    cnt = 0
    for path, dir_list, _ in folders:  
        for dir_name in dir_list:
            folder_path = os.path.join(path, dir_name)
            for file in File_names:
                _, value = read_csv(os.path.join(folder_path, file), config)
                if file == 'temperature.csv':
                    value = clean_temperature(value)

                x.append(value)
                y.append(cnt)
                true_pos.append(dir_name)
            cnt += 1
    print(len(true_pos))
    print(true_pos)
    return x, y, true_pos

def read_colocation_data(building, sensor_count, config):
    x = []  # Store the value
    y = []  # Store the room number
    true_pos = []  # Store the filename
    cnt = 0  # Index for list y
    room_list = []  # Check if there is a sensor in the same room
    room_dict = {}
    groundTruth = read_ground_truth(building)
    # print(groundTruth)
    final_x, final_y, final_true_pos = [], [], []  # output

    # Path depends on where the method is called (where the main is)
    path = "rawdata/metadata/" + str(building)
    # print(path)
    folders = os.walk(path)

    for path, dir_list, files in folders:
        files.remove(".DS_Store")  # We don't want to read in .DS_Store file
        for filename in files:  # Iterating through each time series file
            
            if filename.endswith("csv"):
                # print(filename)
                _, value = read_csv(os.path.join(path, filename), config)
                # Have to clean the data in temperature sensors
                # But I do not know how to retrieve that information in SODA
                # Please change this to fit your code when editing
                if filename == 'temperature.csv':
                    value = clean_temperature(value)

            '''
            Using name as the criteria for same room
            Adding every room and name tuple into a list
            If already in the list, find the corresponding room number
            '''
            if building == "Soda":
                filename = filename.strip(".csv")
                
                if filename[-3:] not in ["VAV", "TMR"]: 
                    '''
                    Excluding VAV because it is not a sensor, rather than control unit
                    Excluding TMR only because we want the four types to be AGN, ASO, ART, and ARS
                    '''
                    # print(filename)
                    find = False  # whether we can find this sensor in groundTruth
                    contains = False
                    # whether this sensor is already contained in one of the rooms represented by elements in y
                    currID = ""

                    # checking whether this sensor is in the groundTruth
                    for currSensor, currRoomID in groundTruth:
                        if currSensor == filename:
                            currID = currRoomID
                            find = True

                    # checking whether this sensor is in an existing room
                    for sensor, tarSensorID, roomNumber in room_list:
                        # print(sensor, tarSensorID, roomNumber)
                        if currID == tarSensorID:
                            contains = True
                            y.append(roomNumber)
                        else:
                            pass

                    if find:
                        if contains:
                            true_pos.append(filename)
                        else:
                            cnt += 1
                            y.append(cnt)
                            true_pos.append(filename)
                            room_list.append([filename, currID, cnt])
                        x.append(value)
            elif building == "SDH":
                if filename.split("+")[-1].split(".")[0] in ["ROOM_TEMP", "AIR_VOLUME", "DMPR_POS"]:
                    room = filename.split("+")[3]
                    if room[0] == "S" and room[2] == "-":
                        if not room in room_dict:
                            room_dict[room] = [filename]
                        else:
                            room_dict[room].append(filename)
                    x.append(value)
                    y.append(room)
                    true_pos.append(filename)
                # print(value)
    # print(room_dict)
    # c = 0
    # for key in room_dict:
    #     print(key, len(room_dict[key]))
    #     c+=1
    # print(c)

    # Only want rooms with specific number of sensors
    # print(y, true_pos)
    # Counting number of sensors in each room
    countDict = {}
    for index in y:
        a = countDict.get(index)
        if a is None:
            countDict.update({index: 1})
        else:
            countDict.update({index: a + 1})
    # print(countDict)
    # Picking rooms with sensor_count sensors
    wantedRoom = []
    indexMap = {}
    roomNum = 0
    for key, value in countDict.items():
        if value == sensor_count:
            # Intuitively we should add the key into the list
            # But the format requires the rooms to be indexes between 0 and length
            # So we create a mapping between real key and the number we want
            wantedRoom.append(key)
            indexMap.update({key: roomNum})
            roomNum += 1

    # Adding desired rooms into output list
    for i in range(len(y)):
        if y[i] in wantedRoom:
            final_x.append(x[i])
            final_y.append(indexMap.get(y[i]))
            final_true_pos.append(true_pos[i])
    # sort lists to fit the format
    zipped_list = zip(final_y, final_x, final_true_pos)
    # print(zipped_list)
    # print("-----")
    # for item in zipped_list:
    #     print("---")
    #     print(item)
    #     print(sorted(item))
    zipped_list = sorted(zipped_list, key=lambda x: x[2])
    final_y = [y for y, x, pos in zipped_list]
    final_x = [x for y, x, pos in zipped_list]
    final_true_pos = [pos for y, x, pos in zipped_list]
    # print(final_x[0])
    print(final_y)
    print(final_true_pos)
    return final_x, final_y, final_true_pos

# def read_in_data(building, config):
#     # read data & STFT
#     x, y, true_pos = read_colocation_data(building, config.sensor_count, config)
#     x = STFT(x, config)
#     return x, y, true_pos


def align_length(ts, val, maxl, sample_f = 5):
    if len(val) >= maxl:
        return ts[0:maxl], val[0:maxl]
    else:
        for i in range(len(val), maxl):
            val.append(0) 
            ts.append(ts[-1] + sample_f)
        return ts, val

def read_csv(path, config):
    f = open(path)
    timestamps, vals = [], []
    for line in f.readlines():
        t, v = line.split(",")
        timestamps.append(int(t))
        vals.append(float(v))
    return align_length(timestamps, vals, config.max_length)

def sub_sample(ts, val, config):
    sample_f = config.interval
    MAXL = config.max_length

    min_ts = ts[0]
    max_ts = ts[-1]
    new_ts, new_val = [], []
    idx = 0
    for t in range(min_ts, max_ts - sample_f, sample_f):
        new_ts.append(t)
        tmp, cnt = 0, 0
        while ts[idx] < t + sample_f:
            tmp += val[idx]
            idx += 1
            cnt += 1
        if tmp != 0:
            new_val.append(tmp / cnt)
        else:
            new_val.append(tmp)
    return align_length(new_ts, new_val, MAXL, sample_f)

def STFT(x, config):
    fft_x = []
    for i in range(len(x)):
        fft_x.append(fft(x[i], config))
    return fft_x

def cross_validation_sample(total_cnt, test_cnt):
    # assert total_cnt % test_cnt == 0

    folds = int(total_cnt / test_cnt)
    idx = list(range(total_cnt))
    random.shuffle(idx)
    test_index = []
    for i in range(folds):
        fold_index = []
        for j in range(test_cnt):
            fold_index.append(idx[test_cnt * i + j])
        test_index.append(fold_index)
    return test_index

def fft(v, config):

    stride = config.stride
    window_size = config.window_size
    k_coefficient = config.k_coefficient
    fft_data = []
    fft_freq = []
    power_spec =[]
    for i in range(int(len(v) / stride)):
        if stride * i + window_size > len(v):
            break
        v0 = v[stride * i: stride * i + window_size]
        v0 = np.array(v0)

        fft_window = np.fft.fft(v0)[1:k_coefficient+1]
        fft_flatten = np.array([fft_window.real, fft_window.imag]).astype(np.float32).flatten('F')
        fft_data.append(fft_flatten)

    return np.transpose(np.array(fft_data))

def split_colocation_train(x, y, test_index, split_method):

    train_x, train_y, test_x, test_y = [], [], [], []
    if split_method == 'room':
        for i in range(len(y)):
            if y[i] in test_index:
                test_x.append(x[i])
                test_y.append(y[i])
            else:
                train_x.append(x[i])
                train_y.append(y[i])
    else:
        for i in range(len(y)):
            if i not in test_index:
                train_x.append(x[i])
                train_y.append(y[i])
            else:
                test_y.append(i)
        test_x = x
    return train_x, train_y, test_x, test_y


def gen_colocation_triplet(train_x, train_y, prevent_same_type = False):
    m = 3
    triplet = []
    for i in range(len(train_x)): #anchor
        for j in range(len(train_x)): #negative
            if prevent_same_type and i % m == j % m: 
                continue
            for k in range(len(train_x)): #positive
                if train_y[i] == train_y[j] or train_y[i] != train_y[k]:
                    continue
                if i == k:
                    continue
                sample = []
                sample.append(train_x[i])
                sample.append(train_x[k])
                sample.append(train_x[j])
                triplet.append(sample)
    return triplet

def gen_coequipment_triplet(ahu_x, train_vav, ahu_y, train_y, mapping):

    triplet = []

    for i in range(len(train_vav)): #anchor
        k = ahu_y.index(mapping[train_y[i]]) #postive
        for j in range(len(ahu_x)): #negative
            if j == k:
                continue
            sample = []
            sample.append(train_vav[i])
            sample.append(ahu_x[k])
            sample.append(ahu_x[j])
            triplet.append(sample)
    ''' 
    for i in range(len(ahu_x)): #anchor
        for j in range(len(train_vav)): #postive
            for k in range(len(train_vav)): #negative
                if mapping[train_y[j]] != ahu_y[i] or mapping[train_y[k]] == ahu_y[i]:
                    continue
                sample = []
                sample.append(ahu_x[i])
                sample.append(train_vav[k])
                sample.append(train_vav[j])
                triplet.append(sample)
    '''
    return triplet
