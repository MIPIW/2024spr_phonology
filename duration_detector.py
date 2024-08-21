#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import json

import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import re
import math
import torch
import argparse
import os
from pathlib import Path
from jamo import h2j, j2hcj
import whisper

from torch import nn
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd


from datasets import load_dataset
from datasets import Audio
import pandas as pd

from datasets import load_dataset
import datasets

def filter_file(file_url):
    with open(file_url, "r") as f:
        out = json.load(f)
    return out


# In[2]:



class IOUtils():
    def __init__(self):
        pass
    
    @staticmethod
    def existsFile(file_url):
        
        import os
        from pathlib import Path

        if type(file_url) == str:
            url = Path(file_url).resolve()
        else:
            url = file_url.resolve()
        
        if url.exists():
            c = 1
            while True:
                if Path(str(url.parent) + "/" + str(url.stem) + f"_{c}" + str(url.suffix)).exists():
                    c += 1
                else:
                    break

            return Path(str(url.parent) + "/" + str(url.stem) + f"_{c}" + str(url.suffix))
        
        print(f"The url of the fIle to be saved is {url}")
        return url
    
    @staticmethod
    def recentFile(file_url):
        
        import os
        from pathlib import Path

        if type(file_url) == str:
            url = Path(file_url).resolve()
        else:
            url = file_url.resolve()

        if url.exists():
            c = 1
            while True:
                if Path(str(url.parent) + "/" + str(url.stem) + f"_{c}" + str(url.suffix)).exists():
                    c += 1
                else:
                    break
        else:
            c = 0
        if c == 0:
            return f"File not exists in {url}"

        if c == 1:
            print(f'The url of the fIle to be loaded is {Path(str(url.parent) + "/" + str(url.stem + str(url.suffix)))}')
            return Path(str(url.parent) + "/" + str(url.stem + str(url.suffix)))
        else:
            print(f'The url of the fIle to be loaded is {Path(str(url.parent) + "/" + str(url.stem + f"_{c-1}" + str(url.suffix)))}')
            return Path(str(url.parent) + "/" + str(url.stem + f"_{c-1}" + str(url.suffix)))

    @staticmethod
    def checkpoint_save(file_path, data, 
                        data_type = "dataFrame", 
                        file_type = "csv", 
                        index_dataFrame = False):
        import json

        save_path = IOUtils.existsFile(file_path)
        
        if data_type == "dataFrame" or data_type == "series":
            import pandas as pd

            if file_type == "csv":
                data.to_csv(save_path, index = index_dataFrame, encoding = 'utf-8')
                                
        
        if data_type == "list":
            if file_type == "txt":
                with open(save_path, "w", encoding = 'utf-8') as f:
                    for item in data:
                        f.write(item)
                        f.write("\n")

                
            if file_type == "jsonl":
                
                with open(save_path, "w", encoding = 'utf-8') as f:
                    for item in data.items():
                        f.write(json.dumps(item))
                        f.write("\n")

        
        if data_type == "dict":
            if file_type == "json":
                
                
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent = "\t")
        
        return data # for the neatness of the code
    
    @staticmethod
    def checkpoint_load(file_path, 
                        data_type = "dataFrame", 
                        file_type = "csv"):
        load_path = IOUtils.recentFile(file_path)
        
        if data_type == "dataFrame" or data_type == "series":
            import pandas as pd

            if file_type == "csv":
                out = pd.read_csv(load_path)
                return out

        if data_type == "list":
            if file_type == "txt":
                out = None

                with open(load_path, "r", encoding = 'utf-8') as f:
                    out = [i.strip() for i in f.readlines()]
                return out

            if file_type == "jsonl":
                out = []

                with open(load_path, "r", encoding = 'utf-8') as f:
                    for line in f:
                        out.append(json.loads(line))

                return out

        
        if data_type == "dict":
            if file_type == "json":
                
                out = None
                
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent = "\t")
                
                with open(load_path, "r") as f:
                    out = json.loads(f)
                
                return out
            


class ParallelizingUtils():
    
    import multiprocessing
    

    def __init__(self):
        self.type = None
        self.func = None
        
        pass
    
    
    def do_series(self, series, num_cores, pre_assign = False):

        from multiprocessing import Pool
        import numpy as np
        import pandas as pd

        if num_cores == 1: ##############somewhere error occurs
            if not pre_assign:
                return self._assign_map(series)
            else:
                return self.func(series)

        se_split = np.array_split(series, num_cores)
        pool = Pool(num_cores)
        if not pre_assign:
            df = pd.concat(pool.map(self._assign_map, se_split))
        else:
            df = pd.concat(pool.map(self.func, se_split))
        pool.close()
        pool.join()
        
        return df
    
    def _assign_map(self, serie):
        return serie.progress_map(self.func)
    
    def _assign_apply(self, df, axis):
        return df.progress_apply(self.func, axis = axis)

    def change_function(self, func):
        self.func = func

    def do_dataFrame(self, df, num_cores, axis = None, pre_assign = False):
        from multiprocessing import Pool
        import numpy as np
        import pandas as pd
        from functools import partial
        
        if num_cores == 1:
            if not pre_assign:
                if axis == None:
                    return valueError("axis needed")
                return self._assign_apply(df, axis = axis)
            else:
                return self.func(df)

        se_split = np.array_split(df, num_cores)
        pool = Pool(num_cores)
        if not pre_assign:
            
            if axis == None:
                    return valueError("axis needed")
            
            f = partial(self._assign_apply, axis = axis)
            df = pd.concat(pool.map(f, se_split))
            
        else:
            df = pd.concat(pool.map(self.func, se_split))
        pool.close()
        pool.join()
        
        return df


class CheckUtils():
    def __init__(self) -> None:
        pass
        import math
    
    # def checkValue(func):
    #     from functools import partial
    #     partial 
    @staticmethod
    def checkSeries(serdf, 
                 isNan = True, 
                 isEmpty = True, 
                 isInf = True,
                    isNotInstance = None,
                    isNotIn = False,
                    isNotInVal = None
                 ):
        

        
        def _check_subsequent(x, 
                              isNotInstance = isNotInstance, 
                              isNan = isNan, 
                              isEmpty = isEmpty, 
                              isInf = isInf, 
                              isnotIn = isNotIn, 
                              isNotInVal = isNotInVal):
            out = False
            
            if isNotInstance is not None:
                x1 = (not isinstance(x, isNotInstance))
            else:
                x1 = False

            if isinstance(x, (int, float, complex)):
                x2 = pd.isna(x) if isNan else False
                x3 = math.isinf(x) if isInf else False
                x4 = False
            
            elif x == None:
                x2 = False
                x3 = False
                x4 = True if isEmpty else False
            
            else:
                x2 = False
                x3 = False
                x4 = (len(x) == 0) if isEmpty else False
                
            x5 = isNotInVal not in x if isNotIn else False

            return out | x1 | x2 | x3 | x4 | x5
        
        
        if isinstance(serdf, pd.Series):
            return serdf.progress_map(lambda x: _check_subsequent(x))
        
        elif isinstance(serdf, pd.DataFrame):
            out  = serdf.map(lambda x: _check_subsequent(x)) # cellwise

            return out.apply(lambda x: any(x), axis = 1)
        
        else:
            raise ValueError("this is not either series or dataframe.")

    
    @staticmethod
    def isEmpty(dataframe) -> bool:
    
        if len(dataframe.index) == 0:
            return True
    
        return False


# In[3]:





############################now unused(original file removed)
# class DataProcess():
#     def __init__(self):
#         pass
    
#     @staticmethod
#     def find_json(dirs):
#         files = os.listdir(dirs)
#         file_list = [file  for file in files if file.endswith("json")]
#         return file_list

#     @staticmethod
#     def filter_json(json_file):
# #         curfolder = "/content/drive/MyDrive/unnamed/"
# #         json_file = Path(curfolder+json_file).resolve()
# #         curfolder = "/content/drive/MyDrive/unnamed/"
#         json_file = Path(json_file).resolve()
#         with open(json_file, "r") as f:
#             out = json.load(f)

#         data = pd.DataFrame(out['document'][0]['utterance'])
#         data = data[data.note.map(lambda x: x == "")]
#         string_yi = re.compile("^[야얘여예요유이]\w*\s|\s[야얘여예요유이]\w*\s|^[야얘여예요유이]\w*$|\s[야얘여예요유이]\w*$")
#         string_ni = re.compile("^[냐냬녀녜뇨뉴니]\w* | [냐냬녀녜뇨뉴니]\w* |^[냐냬녀녜뇨뉴니]\w*$| [냐냬녀녜뇨뉴니]\w*$")
#         string_ri = re.compile("^[랴럐려례료류리]\w* | [랴럐려례료류리]\w* |^[랴럐려례료류리]\w*$| [랴럐려례료류리]\w*$")
#         data_yi = data[data.form.map(lambda x: string_yi.search(x) != None)]
#         data_ni = data[data.form.map(lambda x: string_ni.search(x) != None)]
#         data_ri = data[data.form.map(lambda x: string_ri.search(x) != None)]

#         folder_ser = data_yi.id.map(lambda x: x.split(".")[0])
#         folder_ser.name = "folder"
#         data_yi = pd.concat([folder_ser, data_yi], axis = 1)
        
#         folder_ser = data_ni.id.map(lambda x: x.split(".")[0])
#         folder_ser.name = "folder"
#         data_ni = pd.concat([folder_ser, data_ni], axis = 1)
        
#         folder_ser = data_ri.id.map(lambda x: x.split(".")[0])
#         folder_ser.name = "folder"
#         data_ri = pd.concat([folder_ser, data_ri], axis = 1)

#         return data_yi, data_ni, data_ri

#     @staticmethod
#     def filter_speakers_info(json_file):
#         json_file = Path(json_file).resolve()
#         with open(json_file, "r") as f:
#             out = json.load(f)

#             df = pd.DataFrame(out['document'][0]["metadata"]["speaker"])
#         try:
#             return df[['id', 'sex', 'age', "principal_residence"]]
#         except:
#             x = df[['id', 'sex', 'age']]
#             s = pd.Series(None, name = "principal_residence")
#             return pd.concat([x,s], axis = 1)
        


    # now unused(already converted)
# class SoundDataProcess():
#     def __init__(self):
#         return
    
#     @staticmethod
#     def pcm_to_wav(pcm_list):
#         if self.pcm_list == None:
#             return

#         wav_list = []
# #         curfolder = "/content/drive/MyDrive/unnamed/"
#         for file in pcm_list:
# #             source_file = Path(curfolder+file).resolve()
# #             output_file = Path(curfolder+file).resolve().with_suffix(".wav")
#             source_file = Path(file).resolve()
#             output_file = Path(file).resolve().with_suffix(".wav")
            

#             print(source_file, output_file)

#             # source_file = Path(file).resolve()
#             # output_file = Path(file).resolve().with_suffix('.wav')
#             buf = None

#             with open(source_file, 'rb') as tf:
#                 buf = tf.read()
#                 buf = buf+b'0' if len(buf)%2 else buf    # padding 0 (경우에 따라서 PCM 파일의 길이가 8bit[1byte]로 나누어 떨어지지 않는 경우가 있어 0으로 패딩값을 더해준다, 해당 처리를 하지 않는 경우 numpy나 librosa 라이브러리 사용 시 오류가 날 수 있다)

#             pcm_data = np.frombuffer(buf, dtype='int16')
#             wav_data = lr.util.buf_to_float(x=pcm_data, n_bytes=2)
#             sf.write(output_file, wav_data, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')

#             wav_list.append(output_file.stem)

#         return wav_list
    
#     def find_pcm(dirs):
#         files = os.listdir(dirs)
#         pcm_list = [file  for file in files if file.endswith("pcm")]
#         return pcm_list
 
    

# dataTool = DataProcess()

# x = dataTool.find_file("json")
# year_json = sorted(["json/" + i for i in x[0]])
# year_pcm = sorted(["pcm/" + i for i in x[1]])
    
# out = [dataTool.filter_json(i) for i in year_json]
# yi, ni, ri = [i[0] for i in out],[i[1] for i in out],[i[2] for i in out]
# speaker_info = pd.concat([dataTool.filter_speakers_info(i) for i in year_json])

# df_yi = pd.concat(yi, axis = 0)
# df_ni = pd.concat(ni, axis = 0)
# df_ri = pd.concat(ri, axis = 0)

# speaker_info = pd.concat([dataTool.filter_speakers_info(i) for i in year_json])
# speaker_info = speaker_info[speaker_info.principal_residence.map(lambda x: not pd.isna(x))]
# speaker_info


# df_yi.to_csv("df_yi.csv", index = False)
# df_ni.to_csv("df_ni.csv", index = False)
# df_ri.to_csv("df_ri.csv", index = False)
# speaker_info.to_csv("speaker_info.csv", index = False)


# In[4]:


# move sound file(already processed)
# from shutil import copyfile

# for idx, i in enumerate(df_yi[['folder', 'id']].groupby('folder')):
#     folder = i[0]
#     files_df = i[1].id

#     for ids in files_df:
#         try: 
#             original_path = Path("pcm" + os.sep + folder + os.sep + ids + ".pcm").resolve()
#             to_path = Path("yi" + os.sep + ids + ".pcm").resolve()
#             copyfile(original_path, to_path)
#         except: 
#             pass

# for idx, i in enumerate(df_ni[['folder', 'id']].groupby('folder')):
#     folder = i[0]
#     files_df = i[1].id

#     for ids in files_df:
#         try:
#             original_path = Path("pcm" + os.sep + folder + os.sep + ids + ".pcm").resolve()
#             to_path = Path("ni" + os.sep + ids + ".pcm").resolve()
#             copyfile(original_path, to_path)
#         except: 
#             pass
        
# for idx, i in enumerate(df_ri[['folder', 'id']].groupby('folder')):
#     folder = i[0]
#     files_df = i[1].id

#     for ids in files_df:
#         try: 
#             original_path = Path("pcm" + os.sep + folder + os.sep + ids + ".pcm").resolve()
#             to_path = Path("ri" + os.sep + ids + ".pcm").resolve()
#             copyfile(original_path, to_path)
#         except: 
#             pass


# In[5]:


# convert the pcm file to wav file(already processed)
# lst = ["yi"]
# for i in lst:
#     files = os.listdir(i)
#     files = [i+os.sep+j for j in files]
#     for j in files:
#         target = j
#         destinationPath = j[:-4] + ".wav"
#         print(target, destinationPath)
        
#         buf = None

#         with open(target, 'rb') as tf:
#             buf = tf.read()
#             buf = buf+b'0' if len(buf)%2 else buf    # padding 0 (경우에 따라서 PCM 파일의 길이가 8bit[1byte]로 나누어 떨어지지 않는 경우가 있어 0으로 패딩값을 더해준다, 해당 처리를 하지 않는 경우 numpy나 librosa 라이브러리 사용 시 오류가 날 수 있다)

#         pcm_data = np.frombuffer(buf, dtype='int16')
#         wav_data = lr.util.buf_to_float(x=pcm_data, n_bytes=2)
#         sf.write(destinationPath, wav_data, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
        


# In[6]:


# SNUSLP version

    
class TimeStampModule(nn.Module):
    def __init__(self, module, metadata):
        super(TimeStampModule, self).__init__()
        self.module = module
        self.metadata = metadata
        self.files = list(self.metadata['id_x'])
        
        return
    
    def forward(self, data, filename):
        if ".".join(filename.split(".")[:-1]) in self.files:
            transcript = self.module.transcribe(audio = data,
                                         language = "ko",
                                         word_timestamps = True)
            return filename, transcript
        else:
            return filename, None
        
        

class PrimaryDataLoader(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.lst = None        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        item = self.dataset[idx]
        audio, sr = librosa.load(item['path'], sr = 16000)
        audio = whisper.pad_or_trim(audio)
        return {'filename': item['path'].split("/")[-1], "data": torch.tensor(audio)}
        

class SecondaryDataLoader(Dataset):
    def __init__(self, df, number, listen_more):
        self.df = df
        self.interval = number
        self.listen_more = listen_more
            
        
    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        theline = self.df.iloc[idx]
        audio, sr = librosa.load(theline['path'], sr = 16000)
        left_listen_more = self.listen_more if self.listen_more < int(theline['start']) else int(theline['start'])
        right_listen_more = self.listen_more if self.listen_more + theline['end'] < len(audio) else int(len(audio) - theline['end'])
        
        l = int(theline['start'] - left_listen_more)
        r = int(theline['end'] + right_listen_more)
        seq = audio[l:r]
        original_interval = int(theline['end'] - theline['start'])
        if  original_interval >= self.interval: 
            seq = np.concatenate([seq, np.zeros(self.interval)])
            mask = np.concatenate([np.ones(len(seq)), np.zeros(self.interval)])
        else: 
            seq = np.concatenate([seq, np.zeros(self.interval - original_interval), np.zeros(self.interval)])
            mask = np.concatenate([np.ones(len(seq)), np.zeros(self.interval - original_interval), np.zeros(self.interval)])
        
        
        look_value = int(self.interval + left_listen_more + right_listen_more)
        
        indices = [range(i*16, i*16+self.interval) for i in range(int(look_value/16))]
         
        seq = np.take(seq, indices)
        mask = np.take(mask, indices)
        
        return theline['name'], theline['vowel'], audio, {"input_values": seq, "attention_mask": mask}

class SecondarySubordinateDataLoader(Dataset):
    def __init__(self, value):
        self.value = value
        self.keys = self.value.keys()
    
    def __len__(self):
        return len(self.value["input_values"])
    
    def __getitem__(self, idx):
        return {key: self.value[key][idx] for key in self.value}
    

class PhonemeDetectionModule(nn.Module):
    def __init__(self, model):
        super(PhonemeDetectionModule, self).__init__()
        self.model = model

    def forward(self, data): 
        return self.model(data)
    
    def forward_first(self, data):        
        return self.model(data[0])
    


# In[ ]:





# In[7]:


class MetadataProcessor():
    def __init__(self, speaker_info_url):
        self.speaker_info = pd.read_csv(speaker_info_url).drop_duplicates()
    
    def load_sound_info(self, url_ri, url_ni, types):
        if types == "ri":
            info_df = pd.read_csv(url_ri)
            pattern = re.compile(r'(?<!\wㄴ) ㄹ|^ㄹ\w*')
            
        
        elif types == "ni":
            info_df = pd.read_csv(url_ni)
            pattern = re.compile(r'(?<!\wㄴ) ㄴ|^ㄴ\w*')
            
        else:
            return  
        
        filters = info_df.apply(lambda x: pattern.search(j2hcj(h2j(x['form']))) is not None, axis = 1)
        info_df = info_df[filters]
        info_df = info_df.merge(self.speaker_info, left_on = "speaker_id", right_on = "id")
        
        return info_df


class TimestampDataProcessor():
    def __init__(self, types, metadata):
        types = types
        
        if types == "ri":
            self.word = re.compile("^[랴럐려례료류리]")
        elif types == "ni":
            self.word = re.compile("^[냐냬녀녜뇨뉴니]")
        else:
            self.word = None
        
        self.path = f"/home/hyohyeongjang/phonology/{types}/"
        self.metadata = metadata
        
    def post_module_process(self, filename, transcript_single):
        if transcript_single == None:
            return [(filename, None, None, None, None, None, None)]
        elif len(transcript_single['segments']) == 0:
            return [(filename, None, None, None, None, None, None)]
        else:
            transcript_single = transcript_single['segments'][0]['words']
            # sounds: must exist at least one, could be more than one
            sounds = [(filename, i['word'].strip(), i['start'], i['end']) for i in transcript_single if self.word.match(i['word'].strip())]
            if len(sounds) == 0:
                return [(filename, None, None, None, None, None, None)]
            
            file = ".".join(filename.split(".")[:-1])
            thefile = self.metadata[self.metadata.id_x == file]
            #check if matches
            if len(thefile) >1:
                return [(filename, None, None, None, None, None, None)]

            for i in sounds:
                if i[1] in thefile.form.values[0]:
                    pass
                else:
                    return [(filename, None, None, None, None, None, None)]

            sounds = [i + tuple(thefile[['sex', 'age', 'principal_residence']].values[0]) for i in sounds]
            return sounds
                                              
    def process(self, timestamps, sample_rate):    
        
        def remove_ng(x):
            string = ""
            for i in x:
                y = j2hcj(h2j(i))
                if y[0] == "ㅇ":
                    string += y[1:]
                else:
                    string += y
                    
            return string[0:3]
                
                
        tuples = [item for sublist in timestamps for item in sublist]        
        tuples = pd.DataFrame(tuples)
        filters = CheckUtils.checkSeries(tuples).map(lambda x: not x)
        tuples = tuples[filters]
        tuples.columns = ['name', 'word', 'start', 'end', 'sex', 'age', 'principal_residence']
        tuples['vowel'] = tuples.apply(lambda x: remove_ng(x['word']), axis = 1)
        tuples['start'] = tuples.start * sample_rate
        tuples['end'] = tuples.end * sample_rate
        tuples['path'] = tuples.apply(lambda x: self.path + x['name'], axis = 1)
        
    
        return tuples
    
        
    

class Postprocessor():
    def __init__(self):
        
        self.convert_chart = {"ㅣ": ["I"], 
                              "ㅑ": ["iA"],
                              "ㅕ": ["iEO"],
                              "ㅛ": ["iO"],
                              "ㅠ": ["iU"],
                              "ㅒ": ["iE"],
                              "ㅖ": ["iE"],
                              "ㄹ": ["R", "N"],
                              "ㄴ": ["R", "N"],
                              "ㅂ": ["B","BB","Ph", "M"],
                              "ㅃ": ["B","BB","Ph", "M"],
                              "ㅍ": ["B","BB","Ph", "M"],
                              "ㄷ": ["D","DD", "Th", "N"],
                              "ㄸ": ["D","DD", "Th", "N"],
                              "ㅌ": ["D","DD", "Th", "N"],
                              "ㄱ": ["G","GG","Kh","NG"],
                              "ㄲ": ["G","GG","Kh","NG"],
                              "ㅋ": ["G","GG","Kh","NG"],
                              "ㅅ": ["S","SS","N"],
                              "ㅆ": ["S","SS","N"],
                              "ㅎ": ["H"],
                              "ㅈ": ["J","JJ","CHh","N"],
                              "ㅉ": ["J","JJ","CHh","N"],
                              "ㅊ": ["J","JJ","CHh","N"],
                              "ㅁ": ["M"],
                              "ㅇ": ["NG"],
                              }

    def process(self, result, vowel, name, full_transcription, start_mid_end):
        
        
        if len(vowel) < 3:
            return [None, None, None, name, None, "full_sequence_rejected"]
        else:
#             start_mid_end = [self.convert_chart[i] for i in vowel]
            start_mid_end = start_mid_end.split(" ")
            start_mid_end = ["", start_mid_end[0], start_mid_end[1]]
            
            b = " ".join(start_mid_end[1:])
            c = start_mid_end[2]
            first = full_transcription[0].split(" ")     
            init_consonant = ""
            for j, i in enumerate(first):
                
                if j == 0:
                    continue
                if j == (len(first) - 1):
                    init_consonant = ""
                else:
                    if first[j] == start_mid_end[1] and first[j+1] == start_mid_end[2]:
                        init_consonant = first[j-1]
                        break
                        
            start_mid_end[0] = init_consonant
            if start_mid_end[0] == "":
                a = " ".join(start_mid_end[1:])            
            else:
                a = " ".join(start_mid_end)    
            
            start = np.where(np.array([i.startswith(a) for i in result]) == True)[0]
            mid = np.where(np.array([i.startswith(b) for i in result]) == True)[0]
            end = np.where(np.array([i.startswith(c) for i in result]) == True)[0]

            if any([len(start) == 0, len(mid) == 0, len(end) == 0]):
                return [None, None, None, name, None, "model_not_consistant"]

            start = (min(start), max(start))
            mid = (min(mid), max(mid))
            end = (min(end), max(end))

            out = [0,0,0, name, init_consonant, "cute!"]
            out[0] = start[0]

            if start[1] > mid[0]:
                if start[1] > end[1]:
                    out[1] = mid[0]
                else:
                    out[1] = start[1]
            else:
                out[1] = mid[0]

            if mid[1] > end[0]:
                if mid[1] > end[1]:
                    out[2] = end[0]
                else:
                    out[2] = mid[1]
            else:
                out[2] = end[0]

            return out

    def process_first(self, result, vowel):
        
        
        if len(vowel) < 3:
            return False, ""
        
        else:
#             start_mid_end = [self.convert_chart[i] for i in vowel]
            v1 = self.convert_chart[vowel[0]]
            v2 = self.convert_chart[vowel[1]]
            v3 = self.convert_chart[vowel[2]]
            
            start_mid_end = [f"{j} {k}" for j in v2 for k in v3]
            
            for i in start_mid_end:
                if i in result[0]:
                    return True, i
            
            return False, ""
            
            


# In[8]:


j2hcj(h2j("응"))[1:]


# In[ ]:





# In[9]:


class TemporarySavingModule():
    def __init__(self):
        pass
    
    @staticmethod
    def save_data(data, url_ri, url_ni, types):
        if types == "ri":
            data.to_csv(url_ri, index = False)
        
        elif types == "ni":
            data.to_csv(url_ni, index = False)
            
        else:
            raise ValueError("incorrect types")
        
        return 
    
    @staticmethod
    def read_data(url_ri, url_ni, types):
        if types == "ri":
            data = pd.read_csv(url_ri)
        
        elif types == "ni":
            data = pd.read_csv(url_ni)
            
        else:
            raise ValueError("incorrect types")
        
        return data


# In[ ]:





# In[10]:


# class A():
#     def __init__(self):
#         self.type = None
# args = A()
# args.type = "ni"


# In[7]:


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type = str, required = True)
    parser.add_argument('--froms', type = int, required = True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
      
##############################timestamping##############################


#     # model and metadata
#     metadataProcessor = MetadataProcessor("speaker_info.csv")
#     metadata = metadataProcessor.load_sound_info("df_ri.csv", "df_ni.csv", args.type)
#     print("metadata done")
    
#     timestampModel = whisper.load_model("large")
#     timestampModel.to(device)   
#     timestampModule = TimeStampModule(timestampModel, metadata)
#     print("timestampModule loaded")
    
    
#     # data. bigger than metadata
#     dataset = load_dataset(f"/home/hyohyeongjang/phonology/{args.type}/")
#     dataset = dataset['train']['audio']
#     lst_file = list(metadata.id_x)
#     dataset = [i for i in  dataset if ".".join(i['path'].split("/")[-1].split(".")[:-1]) in lst_file]

#     primary_dataloader = PrimaryDataLoader(dataset)
#     print("dataset loaded")

    
#     timestampProcessor = TimestampDataProcessor(args.type, metadata)
#     # forwarding
#     timestamp = []
#     for j, i in enumerate(primary_dataloader):
#         print(f"{j}st data is under processing...")
#         i['data'] = i['data'].to(device)
#         filename, out = timestampModule(**i)
#         timestamp.append(timestampProcessor.post_module_process(filename, out))
#     print("timestamp checked")
    
#     #postprocessing
#     timestampdata = timestampProcessor.process(timestamp, 16000)
#     TemporarySavingModule.save_data(timestampdata,                            
#                                     url_ri = "secondary_dataloader_ri.csv",                           
#                                     url_ni = "secondary_dataloader_ni.csv",                           
#                                     types = args.type)
#     print("timestamp saved")


# ##############################timestamping##############################
    
    timestampdata = TemporarySavingModule.read_data(url_ri = "secondary_dataloader_ri.csv",                                                               
                                                    url_ni = "secondary_dataloader_ni.csv",                                                               
                                                    types = args.type)
    timestampdata = timestampdata.iloc[args.froms:]
    
    
    print("timestampdata loaded")
    secondary_dataloader = SecondaryDataLoader(timestampdata, 16000, listen_more = 8000)
    
    
#     from transformers import pipeline
#     model_phoneme = pipeline("automatic-speech-recognition", 
#                              model = "slplab/wav2vec2-xls-r-300m_phone-mfa_korean",
#                              device = device
#                             )
#     module_phoneme = PhonemeDetectionModule(model_phoneme)
#     print("phonemedetectionModule loaded")
    
    

    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained("slplab/wav2vec2-xls-r-300m_phone-mfa_korean")
    model = AutoModelForCTC.from_pretrained("slplab/wav2vec2-xls-r-300m_phone-mfa_korean")

    trainingArgs = TrainingArguments("test",
                                    per_device_eval_batch_size = 512)
    trainer = Trainer(model, 
                      trainingArgs,
                      tokenizer = processor.feature_extractor,
                     )
    print("phonemedetectionModule loaded")



    lst = []
    postprocessor = Postprocessor()
    for idx in range(len(secondary_dataloader.df)):
        try:
            print(f"{idx+args.froms}th data is processed")
            name, vowel, original_audio, audio = secondary_dataloader[idx]
            original_audio_dataloader = SecondarySubordinateDataLoader({"input_values": np.array([original_audio]), 
                                                                        "attention_mask": np.array([np.ones(len(original_audio))])
                                                                       })
            out = trainer.predict(test_dataset = original_audio_dataloader)
            predictions = np.argmax(out.predictions, axis = -1)
            full_transcription = processor.batch_decode(predictions)
            truthvalue, midend = postprocessor.process_first(full_transcription, vowel)
            if truthvalue:

                audio_dataloader = SecondarySubordinateDataLoader(audio)
                out = trainer.predict(test_dataset = audio_dataloader)

                predictions = np.argmax(out.predictions, axis = -1)
                out = processor.batch_decode(predictions)
                final = postprocessor.process(out, vowel, name, full_transcription, midend)
                lst.append(final)
            else:
                lst.append([None, None, None, name, None, "full_sequence_rejected"])
        except: 
            lst.append([None, None, None, name, None, "unknown_error"])
        
        if (idx+1) % 100 == 0:
            df = pd.DataFrame(lst)
            TemporarySavingModule.save_data(df,
                                            f"output_ri/start_convert_end_interval_{(idx+args.froms+1)//100}.csv", 
                                            f"output_ni/start_convert_end_interval_{(idx+args.froms+1)//100}.csv",
                                            args.type)
            lst = []
        
    if len(lst) == 0:
        pass
    else:
        df = pd.DataFrame(lst)
        TemporarySavingModule.save_data(df,
                                        f"output_ri/start_convert_end_interval_{((idx+args.froms+1)//100) + 1}.csv",
                                        f"output_ni/start_convert_end_interval_{((idx+args.froms+1)//100) + 1}.csv",
                                        args.type)


# In[ ]:


df2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


timestampdata


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




# target = "/content/drive/MyDrive/unnamed/SDRW2000000559.1.1.1.pcm"
# destinationPath = "/content/drive/MyDrive/unnamed/sample2000000559.1.1.1.wav"
# buf = None

# with open(target, 'rb') as tf:
#     buf = tf.read()
#     buf = buf+b'0' if len(buf)%2 else buf    # padding 0 (경우에 따라서 PCM 파일의 길이가 8bit[1byte]로 나누어 떨어지지 않는 경우가 있어 0으로 패딩값을 더해준다, 해당 처리를 하지 않는 경우 numpy나 librosa 라이브러리 사용 시 오류가 날 수 있다)

# pcm_data = np.frombuffer(buf, dtype='int16')
# wav_data = lr.util.buf_to_float(x=pcm_data, n_bytes=2)
# sf.write(destinationPath, wav_data, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')

