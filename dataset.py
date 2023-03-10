import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from glob import glob
import cv2

class CustomRPPGDataset(Dataset):
    """
    A Custom Datareader for PyTorch

    Args:
        root_dir (str): Path to the root directory of the dataset
        split (str): 'train' or 'val'
        train_ratio (float): Ratio of training data
        data_record_path (str): Path to the data record csv file

    Returns:
        bvps_dict (dict): Dictionary of BVPS
            - bvps_cheek (np.ndarray): BVPS of cheek
            - bvps_forehead (np.ndarray): BVPS of forehead
            - bvps_main (np.ndarray): BVPS of main
        feature_data (dict): Dictionary of feature data
            - bpm (float): BPM
            - ptt (float): PTT
            - sdnn (float): SDNN
            - lfhf (float): LF/HF
            - lf (float): LF
            - hf (float): HF
            - bmi (float): BMI
    """

    def __init__(
            self,
            root_dir:str='dataset',
            split:str='train',
            train_ratio:float=0.8,
            data_record_path:str='data_collection_record.csv',
            ):
        
        self.root_dir = root_dir
        self.data_record_path = data_record_path

        # populate subject
        subjects_name = sorted(os.listdir(root_dir))
        # check if the subject are having the required files
        subjects_name = self.integrity_check(subjects_name)
        # split the subjects
        self.subjects_name = self.split_subjects(subjects_name, train_ratio, split)

    def __len__(self):
        return len(self.subjects_name)
        

    def integrity_check(self, subjects_name):
        exclude_subjects = []
        for name in subjects_name:
            the_path = os.path.join(self.root_dir, name)
            # read the files
            bvps_cheek = np.loadtxt(os.path.join(the_path, 'bvps_cheek.csv'), delimiter=',')
            bvps_forehead = np.loadtxt(os.path.join(the_path, 'bvps_forehead.csv'), delimiter=',')
            bvps_main = np.loadtxt(os.path.join(the_path, 'bvps_main.csv'), delimiter=',')
            feature_data = np.loadtxt(os.path.join(the_path, 'data.csv'), delimiter=',', skiprows=1)

            if len(bvps_cheek) != 1800 or len(bvps_forehead) != 1800 or len(bvps_main) != 1800 or len(feature_data) != 6:
                exclude_subjects.append(name)
                continue

        # remove the subjects
        updated_subjects = [x for x in subjects_name if x not in exclude_subjects]
        if len(exclude_subjects) > 0:
            print('Excluded subjects: ', exclude_subjects)
        return updated_subjects
    
    def split_subjects(self, subjects_name, train_ratio, split):
        if split == 'train':
            subjects_name = subjects_name[:int(len(subjects_name)*train_ratio)]
        elif split == 'val':
            subjects_name = subjects_name[int(len(subjects_name)*train_ratio):]
        else:
            raise ValueError('Invalid split')
        return subjects_name
    
    def get_bmi(self, subject_name):
        # read data record
        data_record = pd.read_csv(self.data_record_path, delimiter=',')
        # name of subject is subject_name but remove last character
        subject_name = subject_name[:-1]
        # find the subject name in the data record column 'ALLIAS' and get data from column 'WEIGHT' and 'HEIGHT'
        weight = data_record[data_record['ALLIAS'] == subject_name]['WEIGHT'].values[0]
        height = data_record[data_record['ALLIAS'] == subject_name]['HEIGHT'].values[0]

        # calculate BMI
        bmi = weight / (height/100)**2
        return bmi
    
    def __getitem__(self, idx):
        name = self.subjects_name[idx]
        the_path = os.path.join(self.root_dir, name)
        # read the files
        bvps_cheek = np.loadtxt(os.path.join(the_path, 'bvps_cheek.csv'), delimiter=',')
        bvps_forehead = np.loadtxt(os.path.join(the_path, 'bvps_forehead.csv'), delimiter=',')
        bvps_main = np.loadtxt(os.path.join(the_path, 'bvps_main.csv'), delimiter=',')
        feature_data = np.loadtxt(os.path.join(the_path, 'data.csv'), delimiter=',', skiprows=1)
        bmi = self.get_bmi(name)

        bvps_dict = {
            'bvps_cheek': bvps_cheek,
            'bvps_forehead': bvps_forehead,
            'bvps_main': bvps_main,
        }

        # feature data only up to 3 decimal places
        feature_data = {
            # 3 decimal places of bpm
            'bpm': round(feature_data[0], 3),
            'ptt': round(feature_data[1], 3),
            'sdnn': round(feature_data[2], 3),
            'lfhf': round(feature_data[3], 3),
            'lf': round(feature_data[4], 3),
            'hf': round(feature_data[5], 3),
            'bmi': round(bmi, 3)
        }
        return bvps_dict, feature_data
    
    
if __name__ == '__main__':
    dataset = CustomRPPGDataset()
    bvps, feature_data = dataset[0]
    print(f"BVPS: {bvps}")
    print(f"Feature Data: {feature_data}")