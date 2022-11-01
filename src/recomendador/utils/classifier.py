# Librerias
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

def train_predict_models(model,X_train, y_train, X_test, y_test, df_salida):

    model.fit(X_train, y_train)            
    y_pred = model.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred,y_test))
    # print(classification_report(y_test,y_pred,target_names=variables))

    return model.predict(df_salida['news_text_content'])

def loggs_results_models(lista_variables, variables, df_salidas, vector_mean):

    promedio_similitud_categorias = {}

    for modelo in lista_variables:

        promedio_similitud_categorias[modelo] = {}

        for categoria in variables:
            indices = df_salidas[df_salidas[modelo]==categoria].index
            vectores_noticias = vector_mean[indices]
            similitudes = cosine_similarity(vectores_noticias,vectores_noticias)
            similitudes_prom = [sim.mean() for sim in similitudes]
            promedio_similitud_categorias[modelo][categoria] = np.array(similitudes_prom).mean()
        
        promedio_similitud_categorias[modelo]["prom_general"] = np.array(list((promedio_similitud_categorias[modelo].values()))).mean()
    
    return promedio_similitud_categorias

def select_best_model(promedio_similitud_categorias):

    mayor = np.zeros(1)
    var_modelo = None

    for modelo in promedio_similitud_categorias.keys():

        nuevo = round(promedio_similitud_categorias[modelo]['prom_general'],3)

        print(f"Promedio general para {modelo} %s" % nuevo)
        if nuevo>mayor:
            mayor = nuevo
            var_modelo = modelo

    print("El mejor modelo con un promedio de similitud de %s contemplando todas las categorias es %s" % (str(mayor), var_modelo))

    return (mayor,var_modelo)

def model_pipelines():
    nb = Pipeline([('vect',CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf',MultinomialNB())])

    sgd = Pipeline([('vect',CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf',SGDClassifier(loss = 'hinge', alpha = 1e-3, random_state=2022, max_iter=5, tol = None))])

    logreg = Pipeline([('vect',CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',LogisticRegression(n_jobs=1, C= 1e5))])
    
    return nb,sgd,logreg


def logisticRegWord2Vec(vector_mean, train, test,df_salida):
    X_train_word_average = vector_mean[train.index]
    X_test_word_average = vector_mean[test.index]

    logreg_w2v = LogisticRegression(n_jobs=1, C= 1e5)
    logreg_w2v = logreg_w2v.fit(X_train_word_average,train.Categoria)

    y_pred = logreg_w2v.predict(X_test_word_average)

    print('accuracy %s' % accuracy_score(y_pred,test.Categoria))
    # print(classification_report(test.Categoria,y_pred,target_names=variables))
    return logreg_w2v.predict(vector_mean[df_salida.index])