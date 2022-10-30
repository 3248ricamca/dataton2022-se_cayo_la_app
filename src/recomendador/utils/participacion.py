import re
from itertools import combinations
from nltk.corpus import stopwords

spanish_stop_words = set(stopwords.words('spanish'))

def check_cliente(cliente, noticia, sector):

    """
        Valida si dentro de un string aparece un texto en concreto.
        Mas espcificamente, esta funcion analiza si dentro de una noticia aparece un cliente.

        Inputs:

            cliente: str

                String con el nombre del cliente.
            
            noticica: str

                String con la noticia

        Outputs: str

            Categorias dependindo de los siguientes criterios

            Si el nombre del cliente es compuesto y tiene menos de 2 palabras, se analiza si el nombre de ese cliente aparece dentro de la noticia o no, buscando directamente ocurrencias simultaneas de ambas palabras. ejemplo: cliente = 'banco republica". Si el cliente es mencionado en la noticia se crea la categoria 'Cliente', de lo contrario 'No Aplica'

            En caso en el que el nombre del cliente tenga mas de 2 palabras ocurren varios casos.

                1. Se generan combinaciones de dos palabras del nombre del cliente y se buscan dentro del texto. Si esa combinacion aparece dentro de la noticia se crea la categoria 'Cliente_'
        
    """

    cliente_edit = cliente.split(" ")

    if len(cliente_edit)<3:

        condicion = bool(re.findall(cliente,noticia))
        if condicion:
            output = 'Cliente'
        else:
            output = 'No Aplica'

    else:
        combs = [ " ".join(list(x)) for x in combinations(cliente_edit, 2)]
        condicion = [bool(re.findall(x,noticia)) for x in combs]

        if True in condicion:
            output = 'Cliente'
            
        else:
            output = "No Aplica"

    if (sector>0.5) and (output == 'No Aplica'):
        output = 'Sector'   

    return output

def clean_name(text,lista_filtro):

    """
    Esta funcion limpia una cadena de texto excluyendo todas aquellas palabras que aparecen dentro de lista_filtro.
    Se usa por defecto la separación de la cadena con espacios en blanco.

    Inputs: 

        text : str

            Texto a limpiar
        
        lista_filtro : list

            Lista de palabras a remover
        
    Outputs:

        String con la cadena de texto limpia

    """
    lista_nombre = text.split(' ')
    nombre_final = [word for word in lista_nombre if word not in lista_filtro and word not in spanish_stop_words]
    return ' '.join(nombre_final)

def elimina_letras_sueltas(text):
    """
        Elimina letras unicas dentro de un texto, por defecto el separador de la cadena de texto son los espacios en blanco.

        Inputs:

            text: str

                String al que se eliminaran las letras suletas

        Outputs: str

            String con el filtrado de caracteres alphabeticos sueltos (De longitud igual a uno)

    """
    text_clean = text.split(" ")
    text_clean = [word for word in text_clean if len(word)>1]

    return " ".join(text_clean)


def clean_df(df, *args, **kwards):

    """
        Esta funcion es auxiliar y permite aplicar funciones previas creadas directamente sobre un dataframe para depurar especificamente campos como el nombre del cliente y generar otro tipo de feature sobre ese nombre.

        Inputs:

            df: pandas.DataFrame
                DataFrame al que se le aplicaran los cambios

            *args and **kwards:
                Argumentos de funciones propias usadas dentro del mismo
        
        Outputs: pandas.DataFrame

            DataFrame de pandas con las correcciones realizadas
    """

    lista_columnas = ['desc_ciiu_division','desc_ciuu_grupo','desc_ciiuu_clase','subsec']

    df_clean = df.copy()
    df_clean['nombre_clean'] = df_clean['nombre'].apply(lambda x: x.replace('.',''))
    df_clean['nombre_clean'] = df_clean['nombre_clean'].apply(lambda x: clean_name(text = x.lower(),*args, **kwards))
    # df_clean['nombre_clean'] = df_clean['nombre_clean'].apply(elimina_letras_sueltas)
    df_clean['nombre_clean'] = df_clean['nombre_clean'].str.title()
    df_clean['long_nombre'] = df_clean['nombre_clean'].apply(lambda x: len(x))
    df_clean['palabras_nombre'] = df_clean['nombre_clean'].apply(lambda x: len(x.split(' ')))


    df_clean['lista_ciiud'] = df_clean[lista_columnas].agg(" ".join, axis = 1)
    df_clean['lista_ciiud'] = df_clean['lista_ciiud'].apply(lambda x: clean_name(x.lower(),[]))
    df_clean['lista_ciiud'] = df_clean['lista_ciiud'].apply(clean_puntuation)

    df_clean = df_clean[['nit',"nombre_clean",'lista_ciiud']]

    return df_clean

def preprocessing_noticas_clientes(x):

    x['Participacion'] = x.apply(lambda x: check_cliente(cliente=x['nombre_clean'], noticia=x['news_text_content']))    
    return x

def noticia_cliente_detalle(noticias,clientes_noticias):

    """

        Esta funcion realiza un merge sobre 2 pandas.DataFrame, posteriormente filtra ciertos campos necesarios. Funcion auxiliar. 

        Inputs:

            noticias: pandas.DataFrame

                Primer df para hacer el merge
            
            clientes: pandas.DataFrame

                Segundo df para hacer el merge
        
        Outputs: pandas.DataFrame

            DataFrame con las variables preseleccioandas sobre el merge de clientes y noticias
        
    """

    final_df = clientes_noticias.merge(noticias, on = 'news_id', how='left')
    final_df = final_df[['nit','news_id', 'news_text_content']]
    return final_df

def clean_puntuation(text):
    """
        Elimina puntuaciones sobre el texto
    """
    text_ = re.sub(r'[^\w\s]', '', text)
    text_ = text_.split(' ')
    text_ = list(set(text_))

    return text_

def conteo_palbras_lista(text, noticia):
    """
        Cuenta cuantas palabras dentro de text son encontradas dentro de una noticia
    """
    contador = [bool(re.findall(word,noticia)) for word in text]
    return contador.count(True)