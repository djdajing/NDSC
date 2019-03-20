import json
import pandas as pd

jsonFiles = ["C:\\fashion.json", "C:\\mobile.json", "C:\\beauty.json"]
csvFiles = ["C:\\fashion_data_info_train_competition.csv", "C:\\mobile_data_info_train_competition.csv",
            "C:\\beauty_data_info_train_competition.csv"]
outputFiles = ["C:\\Users\\user\\desktop\\fashion_data_info_train_competition_edited.csv",
               "C:\\Users\\user\\desktop\\mobile_data_info_train_competition_edited.csv", "C:\\Users\\user\\desktop\\beauty_data_info_train_competition_edited.csv"]
endColArr = [8, 14, 8]
for fileCount in range(len(jsonFiles)):
    motherJSON = {}
    # read in json and remove all spaces special characters in the key values
    with open(jsonFiles[fileCount], encoding='utf-8-sig') as json_file:
        json_data = json.load(json_file)
        for category in json_data:
            bufferJSON = {}
            for key, value in json_data[category].items():
                editedKey = ''.join(e for e in key if e.isalnum())
                bufferJSON[editedKey] = value
            motherJSON[category] = bufferJSON

    # read in csv
    dataframe = pd.read_csv(csvFiles[fileCount], delimiter=',')
    header = list(dataframe.head(0))

    counter = 0
    titleColumn = list(dataframe['title'])
    for title in titleColumn:
        titleColumn[counter] = set(title.split(' '))
        counter += 1

    # [3:8], where 3 is the first element you need and 8 is the first element you don't need
    startCol = 3
    endCol = endColArr[fileCount]
    for attribute in header[startCol:endCol]:
        attributeToFill = dataframe[attribute]
        jsonAttribute = motherJSON[attribute]
        counter = 0
        for cellItem in attributeToFill:
            if pd.isna(cellItem):
                for key, value in jsonAttribute.items():
                    if key in titleColumn[counter]:
                        dataframe.set_value(counter,attribute, value)
                        print('title: ' + str(titleColumn[counter]) + ' row: ' + str(counter + 2) + ' attribute: ' + attribute + ' value: ' + str(value))
                        break
            if counter % 10000 == 0:  # just to check which row it's at
                print(counter)
            counter += 1
    dataframe.to_csv(outputFiles[fileCount], index=False)