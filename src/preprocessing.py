import category_encoders as cat_encoder
import pandas as pd
import numpy as np

class processing:
    def __init__(self, data):
        self.data = data.copy()
        self.dummies = [
            'State',
            'International plan',
            'Voice mail plan',
        ]
        self.drops = [
            'State',
            'International plan',
            'Voice mail plan',
            'Area code',
        ]

    def get_dummies(self):
        dummy_data = self.data[self.dummies]

        # creating an object BinaryEncoder
        encoder = cat_encoder.BinaryEncoder(cols=dummy_data.columns)

        # fitting the columns to a data frame
        df_category_encoder = encoder.fit_transform(dummy_data)
        self.data.drop(self.dummies, axis=1, inplace=True)
        self.data = pd.concat([self.data, df_category_encoder], axis=1)

    def start(self):
        self.get_dummies()
        print('dummies done')