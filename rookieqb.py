#predict best rookie qb performance

import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

#read abbrevations
def processAbbrevations():
    temp = {}
    abbrevs = np.genfromtxt('/users/nikhil/Documents/Football/abbrev_cfb.csv', delimiter=',', dtype=None, encoding=None, usecols=np.arange(0,2))
    for row in abbrevs:
        short = row[1]
        long = row[0]
        #edits to make it compatible with FO data
        if "ORE or" in short:
            short = "ORE"
        if "ALA or" in short:
            short = "ALA"
        if "TAMU" in short:
            short = "TA&M"
        if "OKLA or" in short:
            short = "OKLA"
        if "SC or" in short:
            short = "SC"
        if "MIA" in short:
            short = "MIAMI"
        if "WASH or" in short:
            short = "WASH"
        if "TOL or" in short:
            short = "TOL"
        if "NORF" in short:
            short = "NOR"
        if "MIZ or" in short:
            short = "MIZ"
        if "Cal" in long:
            long = "California"
        if "WKU" in long:
            long = "Western Kentucky"
        if "NC State" in long:
            long = "North Carolina"
        if "ECU" in long:
            long = "East Carolina"
        if "Miami (FL)" in long:
            long = "Miami"
        if "San José State" in long:
            long = "San Jose State"
        if "UCF" in long:
            long = "Central Florida"
        if "FAU" in long:
            long = "Florida Atlantic"
        if "Louisiana Monroe" in long:
            long = "UL-Monroe"
        temp[short] = long
    return temp

#read college QBR
def processCollegeQBR(year):
    temp = np.array([0,0,0])
    qbr = np.genfromtxt('/users/nikhil/Documents/Football/college_qbr/qbr' + str(year) + ".csv", delimiter=',', dtype=None, encoding=None)
    for row in qbr:
        temp = np.vstack((temp, [row[1][1:], row[2], str(row[-1])]))
    return np.delete(temp, (0), axis=0)

#read oline ratings
def processCollegeOline(qbr_array, year):
    oline = np.genfromtxt('/users/nikhil/Documents/Football/college_oline/oline_' + str(year) + ".csv", delimiter=',', dtype=None, encoding=None)
    oline_stats = np.array([0,0,0,0,0,0,0,0,0])
    for row in qbr_array:
        school_abrev = row[1].strip()[:-1]
        fullname = abbrevations[school_abrev]
        index = np.where(oline == fullname)[0][0]
        nums = oline[index][1::2]
        for ind,elem in enumerate(nums):
            if "%" in elem:
                nums[ind] = round(float(elem[:-1]) / 100, 2)
            else:
                nums[ind] = float(elem)
        oline_stats = np.vstack((oline_stats, nums))
    return np.delete(oline_stats, (0), axis=0)

#read defense ratings
def processCollegeDefense(qbr_array, year):
    defense = np.genfromtxt('/users/nikhil/Documents/Football/college_defense/defense_' + str(year) + ".csv", delimiter=',', dtype=str, encoding=None)
    defense_stats = np.array([0]*8)
    for row in qbr_array:
        school_abrev = row[1].strip()[:-1]
        fullname = abbrevations[school_abrev]
        index = np.where(defense == fullname)[0][0]
        nums = np.append(defense[index][3], defense[index][4::2])
        if year >= 2017:
            nums = np.delete(nums, len(nums) - 2, 0)
        defense_stats = np.vstack((defense_stats, nums))
    return np.delete(defense_stats, (0), axis=0)

#read ANY/A stats
def processANYAandOline(qbr_array, year):
    anya = np.genfromtxt('/users/nikhil/Documents/Football/nfl_anya/anya_' + str(year) + ".csv", delimiter=',', dtype=str, encoding=None, usecols=np.arange(0,29))
    anya_stats = np.array([0])
    oline_stats = np.array([0,0])
    for row in qbr_array:
        name = row[0]
        index = np.where(anya == name)
        if (index[0].size == 0):
            anya_stats = np.vstack((anya_stats, -1))
            continue
        anya_stats = np.vstack((anya_stats, float(anya[index[0][0]][-3])))
        team = anya[index[0]][0][2]
        oline_stat = processNFLOline(team)
        oline_stats = np.vstack((oline_stats, oline_stat))
    return np.delete(anya_stats, (0), axis=0), np.delete(oline_stats, (0), axis=0)

#read NFL oline stats team by team - functions differently from other functions
def processNFLOline(team):
    oline_nfl = np.genfromtxt('/users/nikhil/Documents/Football/nfl_oline/nfloline' + str(year) + ".csv", delimiter=',', dtype=str, encoding=None)
    if "NWE" in team:
        team = "NE"
    if "KAN" in team:
        team = "KC"
    index = np.where(oline_nfl == team)
    if len(index) > 1:
        index = max(tuple(map(tuple, index)))
    return [float(x.strip("%")) for x in oline_nfl[index[1]][-2:]]

#generate X and y matrices
def createMasterArrays(qbr, college_oline,nfl_oline, defense, anya):
    X = np.array([0] * 20)
    y = np.array([0])
    oline_ind = 0
    for ind,row in enumerate(anya):
        if row[0] == -1:
            continue
        else:
            X = np.vstack((X, np.append(college_oline[ind], np.append(nfl_oline[oline_ind], np.append(qbr[ind][2:], defense[ind])))))
            y = np.vstack((y,row[0]))
            oline_ind += 1
    return np.delete(X, (0), axis=0),np.delete(y, (0), axis=0)

#constants
abbrevations = processAbbrevations()
X = np.array([])
y = np.array([])

#generate training data
for year in [2014,2015,2016,2017]:
    #preprocess input data
    qbr_master = processCollegeQBR(year)
    college_oline_master = processCollegeOline(qbr_master, year)
    defense_master = processCollegeDefense(qbr_master, year)
    #preprocess output data
    anya_master,nfl_oline_master  = processANYAandOline(qbr_master, year + 1)

    #create one big input and output array
    temp_X,temp_y = createMasterArrays(qbr_master, college_oline_master, nfl_oline_master, defense_master, anya_master)
    if X.shape[0] == 0:
        X = temp_X
        y = temp_y
    else:
        X = np.concatenate((X,temp_X))
        y = np.concatenate((y,temp_y))

#fit model
y=y.ravel()
model = svm.SVR(gamma='auto')
model.fit(X,y)

#check accuracy
scores = cross_val_score(model, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (abs(scores.mean()), scores.std() * 2))
