import pandas as pd
import tensorflow as tf
from keras.utils import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np

lines = pd.read_csv('mysite/languageModel/수어문장.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']

lines.tar = lines.tar.apply(lambda x : '\t '+ x + ' \n')

src_vocab=set()
for line in lines.src: # 1줄씩 읽음
    for char in line: # 1개의 글자씩 읽음
        src_vocab.add(char)
#tar열에서 고유한 글자를 추출해옴
tar_vocab=set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1

src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))

src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])

max_src_len = max([len(line) for line in lines.src])
#new_max_src_len = max([len(line) for line in test_fin_line])
max_tar_len = max([len(line) for line in lines.tar])

index_to_src = dict((i, char) for char, i in src_to_index.items())
index_to_tar = dict((i, char) for char, i in tar_to_index.items())

encoder_model = load_model('mysite/languageModel/test_encoder_model.h5', compile=False)
decoder_model = load_model('mysite/languageModel/test_decoder_model.h5', compile=False)

def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    states_value = encoder_model.predict(input_seq)

    # <SOS>에 해당하는 원-핫 벡터 생성
    target_seq = np.zeros((1, 1, tar_vocab_size))
    target_seq[0, 0, tar_to_index['\t']] = 1.
    

    stop_condition = False
    decoded_sentence = ""

    # stop_condition이 True가 될 때까지 루프 반복
    while not stop_condition:
        
        # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 예측 결과를 문자로 변환
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            continue
        else:
            sampled_char = index_to_tar[sampled_token_index]

        # 현재 시점의 예측 문자를 예측 문장에 추가
        decoded_sentence += sampled_char

        # <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_tar_len):
            stop_condition = True

        # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
        target_seq = np.zeros((1, 1, tar_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1.

        # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
        states_value = [h, c]

    return decoded_sentence

# 단어로 된 문자 배열을 넘겨주면 문장으로 반환
def return_sentence(line):
    sen = ''
    encoder_input = []
    temp_X = []
    for i in line:
        sen += i+' '
    print(type(sen))
    for w in sen: # 각 줄에 있는 문자를 순회합니다.
        temp_X.append(src_to_index[w]) # 각 문자를 'src_to_index' 딕셔너리를 사용하여 정수로 변환합니다.
    encoder_input.append(temp_X) # 변환된 정수의 리스트를 'encoder_input'에 추가합니다.
    encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
    encoder_input = to_categorical(encoder_input, num_classes=52)
    input_seq = encoder_input[0: 0+1]
    decoded_sentence = decode_sequence(input_seq)
    print(35 * "-")
    print('입력 문장:', line)
    print('번역기가 번역한 문장:', decoded_sentence)
    return decoded_sentence

if __name__=="__main__":
    return_sentence(['오늘', '춥다'])