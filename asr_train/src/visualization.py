import librosa
import numpy as np
import scipy.signal as scp

from collections import Counter
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class Visualization:
    def __init__(self, sr: int = 16000):
        self._sr = sr
    
    def plot_waveform(self, text: str, fpath: str):
        signal , _ = librosa.load(fpath, sr=self._sr)
        time = np.linspace(0, len(signal) / self._sr, len(signal))
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name=f"{text} Audio Signal"))

        fig.update_layout(
            title=f"{text} Waveform in Time Domain",
            xaxis_title='Time (Seconds)',
            yaxis_title='Amplitude',
        )
        fig.show()
        return signal

    def plot_amplitude_spectrogram(self, signal: np.array, n_fft: int = 4096):

        sos = scp.butter(4, [85, 255], btype='bandpass', fs=self._sr, output='sos')
        filtered_signal = scp.sosfilt(sos, signal)
       
        D = librosa.stft(filtered_signal, n_fft=n_fft)

        frequencies = librosa.fft_frequencies(sr=self._sr, n_fft=n_fft)
        times = librosa.times_like(D)

        magnitude = np.abs(D)
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=magnitude,               
            x=times,                           
            y=frequencies,                     
            colorscale='RdBu',                
            colorbar=dict(title='Magnitude'),    
        ))

        fig.update_layout(
            title='Dominant Frequencies',
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            showlegend=True,  
            yaxis=dict(range=[0, 300]),
                      
        )

        fig.show()

    def plot_power_spectogram(self, signal: np.array, n_fft: int = 4096):
        
        sos = scp.butter(4, [85, 255], btype='bandpass', fs=self._sr, output='sos')
        filtered_signal = scp.sosfilt(sos, signal)
        filtered_signal = scp.medfilt(signal, kernel_size=5)
       
        D = librosa.stft(filtered_signal, n_fft=n_fft)

        magnitude = np.abs(D)
        power_spectrogram = magnitude ** 2
        power_spectrogram_db = librosa.power_to_db(power_spectrogram, ref=np.max)

        frequencies = librosa.fft_frequencies(sr=self._sr, n_fft=n_fft)
        times = librosa.frames_to_time(np.arange(power_spectrogram_db.shape[1]), sr=self._sr)

        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=power_spectrogram_db,               
            x=times,                           
            y=frequencies,                     
            colorscale='Agsunset',                
            colorbar=dict(title='Power (dB)')
        ))

        fig.update_layout(
            title="Power Spectrogram",
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            showlegend=True,  
            yaxis=dict(range=[20, 200]),
                      
        )

        fig.show()

    
    def plot_duration_hist(self, df):
        fig = go.Figure(data=[go.Histogram(
            x=df['log_duration'],
            nbinsx=60,
            opacity=0.7,
        )])
        
        fig.update_layout(
            title="Duration Distribution of Audio Files (Log duration)",
            xaxis_title="Log Duration (seconds)",
            yaxis_title="Count",
        )
        
        fig.show()

    def plot_duration_boxplot(self, df):
        fig = go.Figure(data=[go.Box(
            y=df['log_duration'],
            boxpoints='outliers',
            name=''

        )])
        
        fig.update_layout(
            title="Duration Boxplot of Audio Files (Log duration)",
            xaxis_title="",
            yaxis_title="Duration (seconds)",
        )
        
        fig.show()

    def plot_spectrogram_power_mean(self, df):
        df.loc[:, 'power_mean'] = df['power_mean'].apply(lambda x: np.abs(x))
    
        fig = px.scatter(df, x='log_duration', y='power_mean', title="Spectrogram Power Mean",
                        labels={'log_duration': 'Log Duration', 'power_mean': 'Power Mean'})
        
        fig.update_layout(
            xaxis=dict(tickmode='linear'),
            yaxis=dict(tickformat='.2f')  # Format y-axis to show 2 decimal places
        )
        
        fig.show()


    def plot_text_score(self, df):
        fig = px.scatter(
                df, 
                x='log_duration', 
                y='power_mean', 
                color='score',
                range_color=[0, 1],
                color_continuous_scale='Agsunset',
                title='Scatter Plot of Votes with Score as Color',
                labels={
                    'up_votes_score': 'Up Votes Score',
                    'down_votes_score': 'Down Votes Score',
                    'score': 'Score'
                },
                opacity=0.6
            )

        fig.update_traces(marker=dict(size=5))
        fig.show()

    def plot_word_length(self, df):
        df['len'] = df['processed_text'].apply(lambda x: len(x))
        fig = px.bar(data_frame=df['len'].value_counts().reset_index(),
             x='len', y='count',
             labels={'len': 'Sentence Length', 'count': 'Frequency'},
             title='Distribution of Sentence Length (by Word)')
        fig.show()
    
    def plot_char_length(self, df):
        df['char_len'] = df['processed_text'].str.replace(' ', "").apply(lambda x: len(x))
        fig = px.bar(data_frame=df['char_len'].value_counts().reset_index(),
              x='char_len', y='count',
             labels={'len': 'Character Length', 'count': 'Frequency'},
             title='Distribution of Sentence Length (by Character)')
        fig.show()

    def plot_word_dist_hist(self, df):
        df['len'] = df['processed_text'].apply(lambda x: len(x))
        fig = px.histogram(
            data_frame=df, 
            x='len', 
            nbins=20,  # Adjust number of bins as needed
            labels={'len': 'Sentence Length (Words)', 'count': 'Frequency'},
            title='Distribution of Sentence Length (by Word)'
        )
        fig.update_layout(yaxis_title="Frequency")
        fig.show()

    def plot_accent_dist(self, df):
        df['accent'] = df['accent'].fillna('others')
        fig = px.bar(data_frame=df['accent'].value_counts().reset_index(),
             x='accent', y='count',
             labels={'accent': 'Accent', 'count': 'Frequency'},
             title='Accent Distribution',
             text='count' 
            )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
    
        fig.show()

    def plot_word_cloud(self, df):
        vocab = " ".join(df['processed_text'])
        tokens = word_tokenize(vocab)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

        word_counts = Counter(filtered_tokens)
        sorted_word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1]))
        threshold = int(len(sorted_word_counts) * 0.2)
        lowest_frequency_words = dict(list(sorted_word_counts.items())[:threshold])

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(lowest_frequency_words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # Turn off the axis
        plt.show()
        return sorted_word_counts


