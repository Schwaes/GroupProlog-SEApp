import os, json, shutil
import librosa
import numpy as np
import pandas as pd

metadata = pd.read_csv('xcmeta.csv', sep='\t', lineterminator='\n')

tags = list(metadata['en'])
all_feats = []
all_tags = []
count = 0

def extract_feat(filepath, index):
	try:
		y, sr = librosa.load(filepath, res_type = 'kaiser_fast')
		mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 40)
		scaled_mfcc = np.mean(mfcc.T, axis = 0)
		all_feats.append(scaled_mfcc)
		all_tags.append(metadata.at[index, 'en'])
	except FileNotFoundError:
		print('file ' + filepath + ' does not exist.')


for index, file_id in enumerate(list(metadata['id'])):
	file_name = 'xc' + str(file_id) + '.wav'
	file_path = os.path.join('flac', file_name)
	extract_feat(file_path, index)
	count += 1
	print(count)



'''
print(len(tags))
print(metadata['id'][:5])
print(len(all_feats))
print(all_feats[:5])
'''

feats_array = np.asarray(all_feats)
tags_array = np.asarray(all_tags)

np.save('feats array.npy', feats_array)
np.save('tags array.npy', tags_array)

