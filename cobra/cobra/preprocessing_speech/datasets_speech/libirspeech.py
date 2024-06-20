import json
import os
import ipdb
import glob
import pickle

def save_pkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    return

def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

train_clean_100_path = '/mnt/hdd/hsyoon/workspace/ES/speech/datasets/LibriSpeech/train-clean-100'
train_clean_360_path = '/mnt/hdd/hsyoon/workspace/ES/speech/datasets/LibriSpeech/train-other-360'
train_other_500_path = '/mnt/hdd/hsyoon/workspace/ES/speech/datasets/LibriSpeech/train-other-500'
dev_clean_path = '/mnt/hdd/hsyoon/workspace/ES/speech/datasets/LibriSpeech/dev-clean'
dev_other_path = '/mnt/hdd/hsyoon/workspace/ES/speech/datasets/LibriSpeech/dev-other'
test_clean_path = '/mnt/hdd/hsyoon/workspace/ES/speech/datasets/LibriSpeech/test-clean'
test_other_path = '/mnt/hdd/hsyoon/workspace/ES/speech/datasets/LibriSpeech/test-other'

# dataset_list = [train_clean_100_path, train_clean_360_path, train_other_500_path]
# dataset_list = [dev_clean_path, dev_other_path]
dataset_list = [test_clean_path, test_other_path]
# =============================================================================================================
# code for the pickle file
# =============================================================================================================
# train_dataset = {'train-clean-100':{'path':[], 'rel_path': [], 'text':[], 'data_id':[]}, 
#                  'train-other-360':{'path':[], 'rel_path': [], 'text':[], 'data_id':[]}, 
#                  'train-other-500':{'path':[], 'rel_path': [], 'text':[], 'data_id':[]}}
# train_dataset = {'dev-clean':{'path':[], 'rel_path': [], 'text':[], 'data_id':[]}, 
#                  'dev-other':{'path':[], 'rel_path': [], 'text':[], 'data_id':[]}}
train_dataset = {'test-clean':{'path':[], 'rel_path': [], 'text':[], 'data_id':[]}, 
                 'test-other':{'path':[], 'rel_path': [], 'text':[], 'data_id':[]}}
# save_file_name = '/mnt/hdd/hsyoon/workspace/samsung/cobra/cobra/data/libirspeech/train.pkl'
# save_file_name = '/mnt/hdd/hsyoon/workspace/samsung/cobra/cobra/data/libirspeech/train.json'
# save_file_name = '/mnt/hdd/hsyoon/workspace/samsung/cobra/cobra/data/libirspeech/eval.pkl'
# save_file_name = '/mnt/hdd/hsyoon/workspace/samsung/cobra/cobra/data/libirspeech/eval.json'
save_file_name = '/mnt/hdd/hsyoon/workspace/samsung/cobra/cobra/data/libirspeech/test.pkl'
# save_file_name = '/mnt/hdd/hsyoon/workspace/samsung/cobra/cobra/data/libirspeech/test.json'

for data_path in dataset_list:
    key = data_path.split('/')[-1]
    folders = glob.glob(os.path.join(data_path, '*','*'))
    
    for folder in folders:
        wav_files = glob.glob(os.path.join(folder, '*.flac'))
        wav_files.sort()
        text_file = glob.glob(os.path.join(folder, '*.txt'))[0]
        text_data = []
        text_data_id = []
        with open(text_file, 'r') as f:
            while True:
                content = f.readline()
                if not content:
                    break
                else:
                    text = content.strip().split(' ')
                    text_data_id.append(text[0])
                    text_data.append(' '.join(text[1:]))
            f.close()

        for wav_file, text, text_id in zip(wav_files, text_data, text_data_id):
            assert text_id in wav_file
            train_dataset[key]['path'].append(wav_file)
            train_dataset[key]['rel_path'].append('/'.join(wav_file.split('/')[-5:]))
            train_dataset[key]['text'].append(text)
            train_dataset[key]['data_id'].append(text_id)
    

save_pkl(train_dataset, save_file_name)

# =============================================================================================================
# code for the json
# =============================================================================================================

# train_dataset = {'train-clean-100':[], 'train-other-360':[], 'train-other-500':[]}
# train_dataset = {'dev-clean':[], 'dev-other':[]}
# train_dataset = {'test-clean':[], 'test-other':[]}
# for data_path in dataset_list:
#     key = data_path.split('/')[-1]
#     folders = glob.glob(os.path.join(data_path, '*','*'))
    
#     for folder in folders:
#         wav_files = glob.glob(os.path.join(folder, '*.flac'))
#         wav_files.sort()
#         text_file = glob.glob(os.path.join(folder, '*.txt'))[0]
#         text_data = []
#         text_data_id = []
#         with open(text_file, 'r') as f:
#             while True:
#                 content = f.readline()
#                 if not content:
#                     break
#                 else:
#                     text = content.strip().split(' ')
#                     text_data_id.append(text[0])
#                     text_data.append(' '.join(text[1:]))
#             f.close()

#         for wav_file, text, text_id in zip(wav_files, text_data, text_data_id):
#             assert text_id in wav_file
#             data_item =  {}
#             data_item['path'] = wav_file
#             data_item['rel_path'] = '/'.join(wav_file.split('/')[-5:])
#             data_item['text'] = text
#             data_item['data_id'] = text_id

#             train_dataset[key].append(data_item)

# save_json(train_dataset, save_file_name)