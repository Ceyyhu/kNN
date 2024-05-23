import pandas as pd
import numpy as np

from collections import Counter

def FeatureNormalization(Data):

    return (Data + 3) / 6

def kNN(TrainData, TestData, Personalities, k):
    TestDataSum =  np.sum(TestData**2, axis=1, keepdims = True)
    TrainDataSum = np.sum(TrainData**2, axis=1, keepdims = True)

    distances = np.sqrt(-2 * TestData.dot(TrainData.T) + TestDataSum + TrainDataSum.T)

    del TestDataSum
    del TrainDataSum
    del TrainData
    del TestData
    
    closest = np.argsort(distances)[:,:k]

    del distances

    closest = Personalities[closest]

    guesses = [Counter(i).most_common(1)[0][0] for i in closest]

    del Personalities
    del closest

    return guesses
      
        
def main(k, FeatureNormal):
    csvdata = pd.read_csv("16P.csv",encoding= "ISO 8859-1")
    csvdata = csvdata.drop(["Response Id"], axis = 1)

    csvdata = csvdata.replace(["ESTJ","ENTJ","ESFJ","ENFJ","ISTJ","ISFJ","INTJ","INFJ","ESTP","ESFP","ENTP","ENFP","ISTP","ISFP","INTP","INFP"],[0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

    Data = csvdata.to_numpy()

    del csvdata

    Personalities = Data[:,-1] 
    
    Data = Data[:,:-1] # Drops the last column
    
    if FeatureNormal:
        print("Feature Normalization: True")
        Data = FeatureNormalization(Data)
    else:
        print("Feature Normalization: False")

    print("k for kNN algorithm : ", k)

    for i in range(1,6):
        
        print(20*"-")
        print("Fold: ", i)
        
        Initial = int(len(Data) * (0.2 * (i-1)) ) # From
        RangeMax = int((len(Data)*((2/10)*i))) # To
    
        TestData = np.array_split(Data,5)[i-1]     # Cross validation 5 fold
        TestData, TestData2 = np.array_split(TestData,2)  # Splitting the Test Data to 2 because using knn for all data at once uses up too much ram

        TrainData = np.concatenate(np.array_split(Data,5)[:i-1] + np.array_split(Data,5)[i:])

        PersonalitiesTest = np.array_split(Personalities,5)[i-1]
        PersonalitiesTrain = np.concatenate(np.array_split(Personalities,5)[:i-1] + np.array_split(Personalities,5)[i:])

        del Initial
        del RangeMax

        guesses = kNN(TrainData, TestData, PersonalitiesTrain, k)
        del TestData
        guesses2 = kNN(TrainData, TestData2, PersonalitiesTrain, k)

        del TrainData
        del TestData2
        del PersonalitiesTrain
        
        guesses.extend(guesses2)

        Classes = {
            0: {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            1: {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            2: {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            3: {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            4: {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            5: {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            6: {"TP": 0, "TN": 0, "FP": 0, "FN": 0}, 
            7: {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            8: {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            9: {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            10:{"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            11:{"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            12:{"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            13:{"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            14:{"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            15:{"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        }

        
        Iter = 0

        for guess in guesses:
            if guess != PersonalitiesTest[Iter]:   # If the guessed personality type is incorrect
                Classes[guess]["FP"] += 1
                Classes[PersonalitiesTest[Iter]]["FN"] += 1
                for result in Classes:
                    if result != guess and result != PersonalitiesTest[Iter]:
                        Classes[result]["TN"] += 1
            else:                                  # If the guessed personality type is correct
                Classes[guess]["TP"] += 1
                for result in Classes:
                    if result != guess:
                        Classes[result]["TN"] += 1
            Iter += 1
  
        del guesses
        del guesses2

        TrueP = 0
        TrueN = 0
        FalseP = 0
        FalseN = 0
        Precision = 0
        Recall = 0

        for i in Classes:
            Precision += Classes[i]["TP"] / (Classes[i]["TP"] + Classes[i]["FP"])
            Recall += Classes[i]["TP"] / (Classes[i]["TP"] + Classes[i]["FN"])
            TrueP += Classes[i]["TP"]
            TrueN += Classes[i]["TN"]
            FalseP += Classes[i]["FP"]
            FalseN += Classes[i]["FN"]
            
        Precision /= 16  # Macro Average
        Recall /= 16     # Macro Average
        
        print("Accuracy: " + str( (TrueP + TrueN) / ( TrueP + TrueN + FalseP + FalseN) ))
        print("Precision Macro Average: ", Precision)
        print("Recall Macro Average: ", Recall)

        del PersonalitiesTest
        del Iter
        del Classes
       