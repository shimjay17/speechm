import json
import ipdb
import os
from tqdm import tqdm

file_path = '/data2/data_account/workspace/data/GigaSpeech/GigaSpeech.json'
save_file_name = '/data2/data_account/workspace/samsung/speech_summarization_mamba/cobra/data/gigaspeech/train_ver3_3min.json'
root_dir = '/data2/data_account/workspace/data/GigaSpeech'

dataset = {'train':[]}

# duration = 180 # 3 minutes(30-)
duration = (30 - 0.333) * 5 + 30 # 3 minutes with 0.333 seconds overlap

def process_text(text):
    text = text.replace(' <COMMA>', ',')
    text = text.replace(' <PERIOD>', '.')
    text = text.replace('<QUESTIONMARK>', '?')
    text = text.replace(' <EXCLAMATIONPOINT>', '!')
    return text

def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)
        
with open(file_path, 'r') as f:
    data = json.load(f)

for audio in tqdm(data['audios']):
    data_item =  {"segment_time": [], 'text': [], 'text_tn': [], 'data_id': []}
    segment_time = 0
    segment_time_check = 0
    for idx, seg in enumerate(audio['segments']):
    # =============================================================================================================
    # data_item =  {}
    # data_item['rlt_path'] = audio['path']
    # data_item['path'] = os.path.join(root_dir, audio['path'])
    # data_item['seg'] = audio['segments']
    # =============================================================================================================
        if '{L}' in seg['subsets']:
            segment_time_check = segment_time + (seg['end_time'] - seg['begin_time'])

            if segment_time_check < duration:
                data_item['rel_path'] = audio['path']
                data_item['path'] = os.path.join(root_dir, audio['path'])
                data_item['segment_time'].append([seg['begin_time'], seg['end_time']])
                data_item['text'].append(process_text(seg['text_tn']))
                data_item['text_tn'].append(seg['text_tn'])
                data_item['data_id'].append(seg['sid'])
                segment_time = segment_time_check

                if idx == len(audio['segments']) - 1:
                    data_item['segment_duration'] = segment_time_check
                    dataset['train'].append(data_item)
                    data_item =  {"segment_time": [], 'text': [], 'text_tn': [], 'data_id': []}
                    segment_time = 0
                    segment_time_check = 0
            
            else:
                data_item['segment_duration'] = segment_time
                dataset['train'].append(data_item)
                data_item =  {"segment_time": [], 'text': [], 'text_tn': [], 'data_id': []}
                segment_time = 0
                segment_time_check = 0
            
        if segment_time_check < duration and idx == len(audio['segments']) - 1 and data_item['segment_time'] != []:
            data_item['segment_duration'] = segment_time_check
            dataset['train'].append(data_item)
            data_item =  {"segment_time": [], 'text': [], 'text_tn': [], 'data_id': []}
            segment_time = 0
            segment_time_check = 0
            


    # dataset['train'].append(data_item)
save_json(dataset, save_file_name)           
    
import ipdb; ipdb.set_trace()