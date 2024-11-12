import re
import os
import sys
import yaml
import librosa
import numpy as np
import pandas as pd
import scipy.signal as scp

class DataPipeline:

    def __init__(self, type: str, group: str, cfg_filepath: str):  
        self._type = type
        self._group = group

        self._cfg = None
        with open(cfg_filepath, "r") as ymlfile:
            self._cfg = yaml.safe_load(ymlfile)

        self._base_path = os.path.join(sys.path[-1],self._cfg['raw_data_path'])
        self._sampling_rate = self._cfg['sampling_rate']  
      
    def show(self):
        df = pd.read_csv(f"{self._base_path}cv-{self._type}-{self._group}.csv", delimiter=',')
        df['group_type'] = df['filename'].apply(lambda x: x.split("/")[0])
        df['filename'] = df[['filename', 'group_type']].apply(lambda x: self._base_path+x['group_type']+"/"+x['filename'], axis=1)
        df = df.drop(columns=['group_type'])
        return df
    
    def process_raw_df(self, df):
        age_map = {"twenties": "teens", "seventies": "senior", "thirties": "middle-aged", "sixties": "senior", "fifties": "senior", "fourties": "middle-aged", "eighties": "senior"}
        df = df.astype({"down_votes": "int", "up_votes": 'int'})
        df['age'] = df['age'].map(age_map)
        return df
    
    def get_gender(self, fname, n_fft=4096):
        signal, _ = librosa.load(fname, sr=self._sampling_rate)
        sos = scp.butter(4, [85, 255], btype='bandpass', fs=self._sampling_rate, output='sos')
        filtered_signal = scp.sosfilt(sos, signal)
        filtered_signal = scp.medfilt(filtered_signal, kernel_size=5)

        D = librosa.stft(filtered_signal, n_fft=n_fft, hop_length=512)
        magnitude = np.abs(D)
        
        dominant_frequencies = np.argmax(magnitude, axis=0)        
        frequencies = librosa.fft_frequencies(sr=self._sampling_rate, n_fft=n_fft)
        
        male_count = np.sum((frequencies[dominant_frequencies] >= 85) & (frequencies[dominant_frequencies] <= 180))
        female_count = np.sum((frequencies[dominant_frequencies] >= 165) & (frequencies[dominant_frequencies] <= 255))
        
        res = None
        if male_count == female_count:
            res = "others"
        elif male_count > female_count:
            res =  "male"
        else:
            res = "female"
        return res
    
    def process_audio(self, df):
        channel_list = []
        duration_list = []
        for filename in df['filename']:
            signal, _ = librosa.load(filename, sr=self._sampling_rate)
            channel_list += [len(signal)]
            duration_list += [signal.shape[0]/self._sampling_rate]
        df['duration'] = duration_list
        df['log_duration'] = df['duration'].apply(lambda x: np.log(x))
        return df
    
    def get_audio(self, df):   
        audio_list = []
        sr_list = [] 
        for filename in df['filename']:
            signal, sr = librosa.load(filename, sr=self._sampling_rate)
            audio_list.append({
                "path": filename,
                "array": np.array(signal, dtype=np.float32),  # Ensure data type is consistent
                "sampling_rate": self._sampling_rate
            })
            sr_list += [sr]
        df['audio'] = audio_list
        return df
    
    def get_spectrogram_fname(self, fname, n_fft = 4096):
        signal, _ = librosa.load(fname, sr=self._sampling_rate)
        sos = scp.butter(4, [85, 255], btype='bandpass', fs=self._sampling_rate, output='sos')
        filtered_signal = scp.sosfilt(sos, signal)
        filtered_signal = scp.medfilt(signal, kernel_size=5)
        D = librosa.stft(filtered_signal, n_fft=n_fft, hop_length=512)
        magnitude = np.abs(D)
        power_spectrogram = magnitude ** 2
        power_spectrogram_db = librosa.power_to_db(power_spectrogram, ref=np.max)

        power_time_mean = np.mean(power_spectrogram_db[20:200,:], axis=0)
        return np.average(power_time_mean)
    
    def process_text(self, df):
        non_alphabetic_or_space_pattern = re.compile("[^a-z' ]")
        df['processed_text'] = df['text'].str.lower().apply(lambda x: re.sub(non_alphabetic_or_space_pattern, '', x))
        df['processed_text'] = df['processed_text'].str.lower()
        return df
        
    def get_confidence_score(self, df):
        alpha = 0.6
        beta = 0.4
        df['up_votes_score'] = df['up_votes'].apply(lambda x: x + 1e-7)
        df['down_votes_score'] = df['down_votes'].apply(lambda x: x + 1e-7)
        df['score'] = df[['up_votes_score', 'down_votes_score']].apply(lambda x: (x['up_votes_score'] * alpha) / ((x['up_votes_score'] * alpha) + (x['down_votes_score'] * beta)), axis=1)
        return df
