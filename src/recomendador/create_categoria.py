#### librerias
import pandas as pd
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split

#### Modulos propios
from utils.embedding import preprocessing_noticias, run_model_word2vec, matrix_by_new, asignacion_categoria
from utils.classifier import train_predict_models, loggs_results_models, select_best_model, model_pipelines, logisticRegWord2Vec

#################################################################################
########################### variables de entorno ################################
#################################################################################

#categorias a predecir
variables = ["macroeconomia","sostenibilidad","regulaciones","reputacion","alianzas","innovacion"]

# Parametros para la ejecucion del modelo Word2Vec
run_model = True
path_model = "/src/recomendador/word2vec.model"
vector_size = 100
cores = cpu_count()
save = False # Modificar este parametro si desea almacenar el modelo

# Matriz de similitud de noticias vs palabras clave
run_matrix = True

# Muestra de Noticias mas similares segun la categoria
muestra_embedding = 100

# Variables de los modelos
lista_variables = ['nb_categoria', 'sgd_categoria', 'logreg_categoria','logreg_w2v_categoria']


if __name__ == '__main__':

    print("Lectura de insumos...")
    noticias = pd.read_csv('datos/noticias.csv')

    print("Preprocesamiento de insumo...")
    data = preprocessing_noticias(noticias)

    print("Iniciando modelado con Word2Vec...")

    model = run_model_word2vec(data[1],run_model,path_model,vector_size,cores,save)
    vocab = list(model.wv.index_to_key)                                     
    print("Modelo creado...")

    print("Iniciando calculo de matriz de similitud de noticias vs categorias...")
    print("Esto puede tomar unos minutos...")
    vector_mean = matrix_by_new(df=data[1], vocab=vocab, model=model, run=run_matrix, vector_size=vector_size)

    print('Asignacion de categorias segun el embedding con la maxima similitud...')
    #Asignacion de categorias segun el embedding
    salida_categorias = asignacion_categoria(df_noticas=data[0],lista_temas=variables,modelo=model,vector_promedio_noticias=vector_mean)
    salida_categorias.sort_values(['Categoria','Similitud'],ascending = False, inplace=True)

    #Muestra de similitudes mas altas segun la muestra
    print('Muestreo de %s mejores noticias similares por categoria para modelo de clasificacion...', muestra_embedding)
    muestra = salida_categorias.groupby('Categoria').head(muestra_embedding)

    print("Iniciando modelado para clasificador...")

    x = muestra['news_text_content']
    y = muestra['Categoria']

    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state=2022)

    # Pipelines a usar
    nb, sgd, logreg = model_pipelines()

    print("Modelando...")

    print("MultinomialNB...")
    salida_categorias['nb_categoria'] = train_predict_models(nb,X_train, y_train, X_test, y_test, salida_categorias)
    print("SGDClassifier...")
    salida_categorias['sgd_categoria'] = train_predict_models(sgd,X_train, y_train, X_test, y_test, salida_categorias)
    print("LogisticRegression...")
    salida_categorias['logreg_categoria'] = train_predict_models(logreg,X_train, y_train, X_test, y_test, salida_categorias)

    # Regresion Logistica con embeddings de Word2Vec
    # transformacion distinta dado que se debe extraer del la matriz de embeddings sobre noticias los vectores promedio
    train, test = train_test_split(muestra, test_size=0.3, random_state = 2022)
    
    print("LogisticRegression sobre Vectores del Embedding...")
    salida_categorias['logreg_w2v_categoria'] = logisticRegWord2Vec(vector_mean,train, test, salida_categorias)

    print("Logs de resultados de modelos...")
    promedio_similitud_categorias = loggs_results_models(lista_variables=lista_variables, variables=variables,df_salidas=salida_categorias, vector_mean=vector_mean)

    print("Seleccion del mejor modelo...")
    similitud_prom, categorias_select = select_best_model(promedio_similitud_categorias=promedio_similitud_categorias)

    df_categoria = salida_categorias[['news_id',categorias_select]].copy()
    df_categoria.rename(columns={"logreg_categoria":"categoria"},inplace=True)

    print("Concatenacion al archivo categorizacion.csv...")
    ## Unir con salida previa del script de participacion
    df_participacion = pd.read_csv('src/data/output/categorizacion.csv')
    df_participacion = df_participacion.merge(df_categoria, on = "news_id", how = 'left')

    ## Escritura del archivo definitivo
    df_participacion.to_csv('src/data/output/categorizacion.csv',index=False)
    print("Resultados escritos con exito!")
