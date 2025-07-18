from optimum.onnxruntime import ORTModelForQuestionAnswering, ORTModelForAudioClassification, ORTModelForSequenceClassification, ORTModelForSpeechSeq2Seq
from transformers import AutoConfig, Wav2Vec2Processor, AutoTokenizer, PretrainedConfig, WhisperProcessor

import time
import librosa
import numpy as np
import pandas as pd

import os
import re

import metric
import config

class  BaseModel():
    def __init__(self, model_name = ''):
        
        self.model =  ort.InferenceSession(model_name)
    def preprocess(self, input):
        return input
        
    def run(self, inputs):
        return self.model.run(output_names=[ "output" ], input_feed=dict(inputs))

    def postprocess(self, output):
        return output


# кучь в текст
class  SpeechtoText(BaseModel):
    def __init__(self, model_path, model_name, model_name_preprocess, postprocess = None):
        self.processor =  WhisperProcessor.from_pretrained(model_name_preprocess)#"openai/whisper-tiny")
        # self.sampling_rate = processor.feature_extractor.sampling_rate
        model_config = PretrainedConfig.from_pretrained(model_name)
        predictions = []
        references = []
        sessions = ORTModelForSpeechSeq2Seq.load_model(
            os.path.join(model_path, 'encoder_model.onnx'),
            os.path.join(model_path, 'decoder_model.onnx'),
            os.path.join(model_path, 'decoder_with_past_model.onnx'))
        self.model = ORTModelForSpeechSeq2Seq(sessions[0], sessions[1], model_config, model_path, sessions[2])
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="russian", task="transcribe")
        if  postprocess is not None:
            self.post_process =  postprocess
        else:
            self.post_process = None
        self.sampling_rate = 16000
    def postprocess(self, output):

        output = self.processor.batch_decode(output, skip_special_tokens=True)
        return output
        
    def run(self, speech):
        # speech # речевой отрывок
        feature = self.processor(speech, sampling_rate=self.sampling_rate, return_tensors="pt")
        rez = self.model.generate(feature['input_features'], forced_decoder_ids=self.forced_decoder_ids)
        # print(rez.logits[0], self.emotions)
        return self.postprocess(rez)


# речь в эмоции
class  SpeechtoEmotion(BaseModel):
    def __init__(self, model_name, model_name_preprocess, model_postprocess = None):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name_preprocess)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        self.model = ORTModelForAudioClassification.from_pretrained(model_name)#"wav2vec2-xls-r-300m-emotion-ru_onnx"
        self.emotions = ['neutral', 'positive', 'angry', 'sad', 'other']
        if  model_postprocess is not None:
            self.post_process =  model_postprocess
        else:
            self.post_process = None
    
    def postprocess(self, outputs):
        return self.emotions[np.argmax(outputs.logits[0])]
        
    def run(self, speech):
        # speech # речевой отрывок
        features = self.processor(speech, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        rez = self.model(features['input_values'])
        # print(rez.logits[0], self.emotions)
        return self.postprocess(rez)


# текст в эмоции
class  TexttoEmotion(BaseModel):
    def __init__(self, model_name, model_name_or_path):
        self.processor = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = ORTModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ['neutral', 'joy', 'sadness', 'anger', 'enthusiasm', 'surprise', 'disgust', 'fear', 'guilt', 'shame']
        self.labels_ru = ['нейтрально', 'радость', 'грусть', 'гнев', 'интерес', 'удивление', 'отвращение', 'страх', 'вина', 'стыд']
    
    def postprocess(self, outputs):
        return self.labels[np.argmax(outputs.logits[0])]
        
    def run(self, text):
        # text # текстовй отрывок
        
        features = self.processor(text,  max_length=512, truncation=True, return_tensors='pt')
        rez = self.model(**features)
        # print(features )
        return self.postprocess(rez)


# суммарная эмоция
def sum_emo(list_emo):
    rez_class = 0
    for i, name in enumerate(list_emo):
        rez_class += config.emo_class[name] * config.weigth_model[i]
    return rez_class   


def main_audio(model_text, model_textem, model_em, speech=None, dt=1, t=0):
    '''
    model_text=None, # модель текстовой обработки речи
    model_textem=None, # модель обработки текст в эмоции
    model_em=None # модель речь в эмоции
    speech=None,  # массив звука  
    dt=1, # дельта времени для сдвига паудио по времени
    t=0 # время/ 
    '''
    if speech is not None:
        speech_d = speech
        rez_text = model_text.run(speech)
        # print(rez_text)
        rez_word = rez_text[0].split()[0]

        rez_em_text = model_textem.run(rez_text)
    
        
        rez_em_speech = model_em.run(speech)
        # запись результата : время, текст, слово для привязки времени, эмоция по тексту, Эмоция по звуку
        rez_emo = sum_emo([rez_em_text, rez_em_speech])
        rez_ = [t * float(dt) / 2, rez_text[0], rez_word , rez_em_text, rez_em_speech, rez_emo]
        # print('Time: ', t1-t0)
        # print('Result text: ',rez_text)
        # print('Result emotion: ', rez_em_text, rez_em_speech)
        return rez_
    else:
        return [t * float(dt) / 2, '', '' , None, None]


# сборка ответа в строку
def concat_drop(df, plot_rez = 1):
    """
    df : dataFrame $ (t 	sub_string 	word 	em1 	em2 	emotion 	k2 	k1)
    plot_rez = 1 (1 - показывать вывод, 0 -  не показывать)

    return:
     df - с полями k2 	k1
     s - строка собрана из позстрок
    
    """
  
    s1 = df.sub_string.values[0].lower()
    l1 = s1.split(' ')
    
    l1f = [s for s in l1 if len(s)>3 ]
    l1fi = [1 if len(s)>4 else 0 for s in l1]
    s = s1
    # matches = re.finditer(' ', s1)
    # indices1 = [match.start() for match in matches]
    
    for ik, s2 in enumerate(df.sub_string.values[1:].tolist()):
        
        ik = ik + 1
        s2 = s2.lower()
        l2 = s2.split(' ')
        l2f = [s for s in l2 if len(s)>3 ]
        l2fi = [1 if len(s)>4 else 0 for s in l2]
        list_compare = []
        # matches = re.finditer(' ', s2)
        # indices2 = [match.start() for match in matches]
        
        for word in l2f:
            jaro_score = lambda x: metric.jellyfish.jaro_similarity(x, word)
            list_compare += [list(map(jaro_score, l1f))]
        if plot_rez:
            print(list_compare)    
        k = np.argmax(list_compare)
        m = np.max(list_compare)
        
        if m>0.8:
            i, j = k%len(l1f), k//len(l1f)
            
            k1 = s1.find(l1f[i])
            k2 = s2.find(l2f[j]) 
            
            df['k2'].values[ik-1] = k1 
            df['k1'].values[ik] = k2-1
            df['k2'].values[ik] = len(s2)
            df['word'].values[ik] = l2f[j]
            
        s1 = s2
        l1 = l2.copy()
        l1f = l2f.copy()
        l1fi = l2fi.copy()
        
    s = ''
    for i in range(df.shape[0]):
        k1  = int(df.k1.values[i])
        k2  = int(df.k2.values[i])
        
        s = s + df.sub_string.values[i][k1:k2] 
      
    return s, df  


def check_model(path_a, dt=1, d=16000, n_d=4, sampling_rate = 16000,   model_text=None, model_textem=None, model_em=None):
    """
    Paramer:
        
        path_a, # путь к файлу аудио
        dt=1, # дельта времени для сдвига паудио по времени
        d=16000, # длина примеров за dt в точках
        n_d=4, # перекрытие 
        sampling_rate = 16000,   # число точек в с. (частота дискретизации)
        model_text=None, # модель текстовой обработки речи
        model_textem=None, # модель обработки текст в эмоции
        model_em=None # модель речь в эмоции
    Return:
        s : строка ответа 
        df : таблица с разметкой (t: float - веремя от начала фрагмента аудио в с, sub_sring: текст из фрагмета аудио, word:string - первое слово текстового отрезка для разметки,  ) 
        
    """
    if (model_text is not None) and (model_textem is not None) and (model_em is not None):
        try:

            # **********************************************
        
            speech, sr = librosa.load(path_a, sr=sampling_rate)
            # *****************************************************
            rez_all_result = []
            # print('audio ok')

            # распознавание
            t0 = time.time()
            for t, speech_i in enumerate(range(0,speech.shape[0], d)):
                t1 = time.time()
                speech_d = speech[speech_i:speech_i+d * n_d]
                rez_ = main_audio(model_text, model_textem, model_em, speech=speech_d, dt=1, t=t*dt)
                # запись результата : время, текст, слово для привязки времени, эмоция по тексту, Эмоция по звуку
                rez_all_result.append(rez_)    
                # print('Time: ', t1-t0)
                # print('Result text: ',rez_)
            
            df = pd.DataFrame(rez_all_result , columns= ['t', 'sub_string', 'word', 'em1', 'em2','emotion'])#.to_csv(path_a.split('/')[-1].split('.')[0] +  '.csv')
            df['k2'] = 0
            df['k1'] = 0
            
            s, df = concat_drop(df, plot_rez = 0)
            return s, df
        except :
            print('no audio or no convert')
            return '', None
    else:
        print('no model')
        return '', None 
    