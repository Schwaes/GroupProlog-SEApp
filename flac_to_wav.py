from pathlib import PurePath
from pydub import AudioSegment
import os, json, shutil
import numpy as np
import pandas as pd
import soundfile as sf

os.chdir('C:/Users/benja/OneDrive/TiU/5th Semester/Software engineering/Birdsound classification/flac')

count = 0

metadata = pd.read_csv('C:/Users/benja/OneDrive/TiU/5th Semester/Software engineering/Birdsound classification/xcmeta.csv', sep='\t', lineterminator='\n')

dataset = []

files = [f for f in os.listdir('.') if os.path.isfile(f)]
for index, f in enumerate(files):
	audio, sr = sf.read(f)
	file_id = f.removesuffix('.flac')
	wav_id = file_id + '.wav'
	sf.write(wav_id, audio, sr, 'PCM_16')
	count += 1
	print(count)
