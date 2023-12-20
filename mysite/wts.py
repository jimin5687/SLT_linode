import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('jhgan/ko-sroberta-multitask')

df = pd.read_excel("D:/SLT_Website_Django/SLT_linode/mysite/tt_text.xlsx")
df['embedding'] = pd.Series([[]] * len(df))

df['embedding'] = df['src'].map(lambda x: model.encode(x))
df.to_csv('WtoS.csv', index=False)

def translator(line):
    sen = ''
    encoder_input = []
    temp_X = []
    for i in line:
        sen += i+' '

    embedding = model.encode(sen)
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # print('구분', answer[''])
    return answer["tar"]

if __name__=="__main__":
    sentence = translator(["나", "배부르다", "우유"])
    print(sentence)