import pandas as pd 
import os 
import numpy as np 
from sklearn.preprocessing import LabelEncoder

class Encoder(object):
    def __init__(self, data : pd.DataFrame) -> None:
        self.sex_le = LabelEncoder()
        self.smoker_le = LabelEncoder()
        self.reg_le = LabelEncoder()

        self.sex_le.fit(data.sex.drop_duplicates()) 
        self.smoker_le.fit(data.smoker.drop_duplicates()) 
        self.reg_le.fit(data.region.drop_duplicates()) 

    def transform(self, data : pd.DataFrame):
        data.sex = self.sex_le.transform(data.sex)
        data.smoker = self.smoker_le.transform(data.smoker)
        data.region = self.reg_le.transform(data.region)

    def rev_transform(self, data : pd.DataFrame):
        data.sex = self.sex_le.inverse_transform(data.sex)
        data.smoker = self.smoker_le.inverse_transform(data.smoker)
        data.region = self.reg_le.inverse_transform(data.region)




