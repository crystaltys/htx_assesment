import os
import io
import yaml
import logging
import pandas as pd
from pydub import AudioSegment

from src.interface.payload import AsrPayload
from src.services.asr_api import Inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPipeline(Inference):
    def __init__(self, cfg_filepath: str):
        self._cfg = None
        with open(cfg_filepath, "r") as ymlfile:
            self._cfg = yaml.safe_load(ymlfile)
        
        _model_name = self._cfg['model']['selector']
        super().__init__(_model_name)
        
        self._type = self._cfg['data_type']
        self._group = self._cfg['data_group']
        self._base_path = self._cfg['raw_data_path']
        self._sampling_rate = self._cfg['model'][_model_name]['frequency']

       
    def run_batch(self):
        df = pd.read_csv(f"{self._base_path}cv-{self._type}-{self._group}.csv", delimiter=',')
        df = df.sample(n=1000, random_state=42)
        processed_df = self.process_raw_data(df)
        out_arr = self.predict(processed_df)
        df['generated_text'] = [data.transcription if data.transcription else "" for data in out_arr ]
        df.to_csv(f"{self._type}_{self._group}.csv", index=False) 
    
    def run(self, payload: AsrPayload):
        df = pd.DataFrame([payload.file], columns=['filename'])
        out_sample = self.predict(df)[-1]
        if out_sample:
            logging.DEBUG(f"Mock removing file")
            #os.remove(payload.file)
        return out_sample

    def process_raw_data(self, df: pd.DataFrame):
        df['group_type'] = df['filename'].apply(lambda x: x.split("/")[0])
        df['filename'] = df[['filename', 'group_type']].apply(lambda x: self._base_path+x['group_type']+"/"+x['filename'], axis=1)
        return df
    
    def predict(self, df):
        try:
            out_data = []
            for i, (_, row) in enumerate(df.iterrows()):
                if not os.path.exists(row['filename']):
                    raise FileNotFoundError
                audio = AudioSegment.from_mp3(row['filename']).set_frame_rate(self._sampling_rate)
                mp3_io=io.BytesIO()
                audio.export(mp3_io, format="mp3")
                out_data += [self.query(mp3_io, audio.duration_seconds)]
                logging.info(f"Starting inference for data {i+1, row['filename']} output ...")
        except FileNotFoundError as e:
            print(f"FileNotFound: {row['filename']}")
            raise
        return out_data


if __name__ == "__main__":
    CONFIG_PATH = "conf/app_config.yml"
    decoder = DataPipeline(CONFIG_PATH)
    decoder.run_batch()