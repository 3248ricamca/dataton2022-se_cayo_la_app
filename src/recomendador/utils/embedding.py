import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.matutils import unitvec
from gensim.models import Word2Vec


spanish_stop_words = set(stopwords.words('spanish'))

def clean_text_news(df):
    df['len_titular'] = df['news_title'].apply(lambda x: len(x))
    df['len_content'] = df['news_text_content'].apply(lambda x: len(x))
    return df
    
def create_tokenization(x):
    data = []
    for j in word_tokenize(x,language='spanish'):
        term = j.lower()

        if term.isalpha() and not term in spanish_stop_words:
            #term = snowballstemmer.stem(term)
            data.append(term)
        
    return data

def document_vector(word2vec_model, doc, vocab):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in vocab]
    return np.mean(word2vec_model.wv[doc], axis=0)

# Function that will help us drop documents that have no word vectors in word2vec
def has_vector_representation(vocab, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in vocab for word in doc)

# Filter out documents
def filter_docs(corpus, texts, condition_on_doc,vocab):
    """
    Filter corpus and texts given the function condition_on_doc which takes a doc. The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if has_vector_representation(vocab,doc)]

    corpus = [doc for doc in corpus if has_vector_representation(vocab,doc)]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts)

def similitud(v1,v2):
    """
        Esta funcion calcula la similitud de dos vectores de misma longitud

        Inputs:
            v1: vector 1
            v2: vector 2
        
        Output: float

            Similitud de vectores

    """
    return np.dot(unitvec(v1), unitvec(v2))

def asignacion_categoria(df_noticas,lista_temas,modelo,vector_promedio_noticias):

    diccionario_resultados = {}
    for variable in lista_temas:
        vector_tema = modelo.wv[variable]
        similitud_noticias = [similitud(a,vector_tema) for a in vector_promedio_noticias]
        diccionario_resultados[variable] = similitud_noticias

    resultados_temas = pd.DataFrame(diccionario_resultados)
    resultados_temas['Categoria'] = resultados_temas.idxmax(axis=1)
    resultados_temas['Similitud'] = resultados_temas.max(axis=1)

    # df_salida = pd.concat([df_noticas,resultados_temas[['Categoria']]],axis=1)
    df_salida = pd.concat([df_noticas,resultados_temas[['Categoria','Similitud']]],axis=1)

    return df_salida

def preprocessing_noticias(df):

    print(df.shape)
    df = df[df['news_text_content']!=' '].reset_index(drop=True).copy()
    print(df.shape)
    df = df.drop_duplicates()
    print(df.shape)
    data = df['news_text_content'].apply(create_tokenization)
    
    return [df, data]

def matrix_by_new(df,model, vocab, vector_size, output_name = None, run = False, save = False):

    if run:
        new_mean_vector = []
        for doc in df: # append the vector for each document
            try:
                output = document_vector(model, doc, vocab=vocab)
            except:
                output = np.zeros(vector_size)
            new_mean_vector.append(output)

        mean_vector = np.array(new_mean_vector) # list to array
        if save:
            with open(output_name+'.npy', 'wb') as f:
                np.save(f, new_mean_vector)

    else:
        mean_vector = np.load('array_mean_vector_model.npy')
        
    return mean_vector

def run_model_word2vec(datos,run_model,path_model,vector_size,cores,save):

    if run_model:
        model = Word2Vec(datos,min_count=1,
                        window=5,
                        vector_size=vector_size,
                        workers=cores-1,
                        sg = 1)
        if save:
            model.save(path_model)

    else:
        model = Word2Vec.load(path_model)