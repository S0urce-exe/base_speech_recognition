# частота записей
sampling_rate =  16000

# длина эпизода записи в с. 
dt = 1 # длина отрывка в с
d = 16000 * dt # длина отрывка в числе точек
n_d = 4 # коэффициент перекрытия
# имена моделей

dir_model = ''

model_name_speech_to_text = [dir_model+'whisper-tiny_onnx', "openai/whisper-base", "openai/whisper-tiny"]
model_name_speech_to_emo = [dir_model+"wav2vec2-xls-r-300m-emotion-ru_onnx", "KELONMYOSA/wav2vec2-xls-r-300m-emotion-ru"]
model_name_text_to_emo = [dir_model+"rubert-tiny2-russian-emotion-detection_onnx", "Djacon/rubert-tiny2-russian-emotion-detection"]

class_list = {0:['enthusiasm', 'happiness', 'neutral','positive', 'surprise'], 1:['angry', 'disgust', 'fear', 'sadness']}
emo_class = {'positive':0, 'surprise':0.3, 'neutral':0,'enthusiasm':0.1, 'happiness':0, 'joy':0.5,'angry':1, 'anger':1, 'disgust':0.5, 'fear':0.3, 'sadness':0.8}
weigth_model = [0.5, 0.5]
# resd
# {'anger', 'disgust', 'enthusiasm', 'fear', 'happiness', 'neutral', 'sadness'}
