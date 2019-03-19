import pandas as pd


class CsvHelper:
    def __init__(self):
        self.csv = None
        self.category = None
        self.label_headers =None

    def set_category(self,cat):
        self.category = cat

    def get_category (self):
         return self.category

    def get_csv(self):
        return self.csv

    def add_column(self,column_name):
        if column_name in self.csv.columns: #if column exist, remove
            self.csv = self.csv.drop(column_name,axis=1)
        self.csv[column_name]=""

    def set_csv(self,csv_path):
        self.csv = pd.read_csv(csv_path, index_col=None)

    def get_single_column(self, header):
        return self.csv[header]

    def get_id_title_and_single_column(self, header):
        return self.csv.dropna(subset=[header])[['itemid','title',header]]

    def get_title_and_single_column(self, header):
        return self.csv.dropna(subset=[header])[['title',header]]

    def get_id_title(self):
        return self.csv[['itemid', 'title']]

    def get_header(self):
        return list(self.csv)

    def set_label_headers(self, start, end):
        if (end ==-1):
            end = len(list(self.csv))
        self.label_headers= list(self.csv)[start-1:end]

    def set_all_headers_as_label(self):
        all_header_size = len(list(self.csv))
        self.label_headers = list(self.csv)[3: all_header_size]

    def get_label_headers(self):
        return self.label_headers

    def gen_img_path(self,img_name,category):
        return category+"/"+img_name

    def get_label_by_img_name(self,img_name):
        labels = []
        img_path = self.gen_img_path(img_name, self.category)
        item_row = self.csv.loc[self.csv["image_path"]==img_path]
        for label in item_row :
            if label in self.label_headers:
                labels.append(item_row[label].values[0].item())#.value to get the numpynarray [0] to get the actaul value .item to convert it to native python type float

        return labels
#
# category = "mobile_image"
# csv_path ="/home/dj/NDSC/scripts/sample_mobile.csv"
#
# test = CsvHelper()
# test.set_csv(csv_path)
# label_headers = test.get_header_by_range(4, -1)
#
# img_name = "90e03825cee952859bf9352227215b25.jpg"
# print test.get_label_by_img_name(img_name, label_headers,category)