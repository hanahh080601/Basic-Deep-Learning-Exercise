from config import *

class DataPipeline:
    """
    Class DataPipepline
        Preprocessing data, including: reading, getting and splitting data for modeling tasks.
    ---------------------
    Properties:
        source: str
        Path of source containing raw data needed to process.
    ---------------------
    Methods:
        __init__(self, source: str):
        Initialize path to source data.
        
        read_data(self):
        Read .npy file from source to an np.array for modeling tasks.
        
        get_data(self, mode='train'):
        Get necessary features from raw data.
    """
    def __init__(self, source='dataset'):
        self.source = source

    def read_data(self):
        self.train = pd.read_csv(self.source + "/train_record.csv").to_numpy()
        self.test = pd.read_csv(self.source + "/test_record.csv").to_numpy()
        self.X_train = self.train[:, :-1]
        self.y_train = self.train[:, -1]
        self.X_test = self.test[:, :-1]
        self.y_test = self.test[:, -1]

    def preprocess_data(self, scale_mode):

        scaler = MinMaxScaler()
        if scale_mode == 'robust':
            scaler = RobustScaler()
        elif scale_mode == 'standard':
            scaler = StandardScaler()

        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def get_data(self, mode='train'):
        if mode == 'train':
            return self.X_train, self.y_train
        else:  
            return self.X_test, self.y_test
        
    def __call__(self, mode='train', scale_mode='minmax'):
        self.read_data()
        self.preprocess_data(scale_mode=scale_mode)
        return self.get_data(mode)