# librerias de python
import json
import pandas as pd

# modulo propio
from utils.participacion import clean_df, noticia_cliente_detalle, clean_name, conteo_palbras_lista, check_cliente


if __name__ == '__main__':

    print("Lectura de insumos...")
    # Importar lista de siglas para filtrado del nombre de empresas
    with open("src/data/archivos_auxiliares/diccionario_siglas.json", 'r') as f:
        diccionario_siglas = json.load(f)
        lista_siglas = diccionario_siglas['siglas']
    
    clientes = pd.read_csv('datos/clientes.csv')
    noticias = pd.read_csv('datos/noticias.csv')
    clientes_noticias = pd.read_csv('datos/clientes_noticias.csv')

    print("Limpieza de datos...")

    # Limpieza del noticias
    clientes_clean = clean_df(clientes, lista_filtro = lista_siglas)
    noticia_cliente_detalle_df = noticia_cliente_detalle(noticias,clientes_noticias)

    # merge para el dataset final
    participacion_df = clientes_clean.merge(noticia_cliente_detalle_df, on = 'nit', how = 'left')

    # Quitar stopr words de noticas para buscar nombre en mayusc de la empresa
    participacion_df['cuerpo_not_clean'] = participacion_df['news_text_content'].apply(lambda x: clean_name(text = x, lista_filtro=[]))

    print("Calculo de variable participacion...")

    participacion_df["Sector_indice"] = participacion_df.apply(lambda x: conteo_palbras_lista(text=x['lista_ciiud'], noticia=x['cuerpo_not_clean'].lower()), axis=1)    
    participacion_df["Conteo_Ciiu"] = participacion_df['lista_ciiud'].apply(len)
    participacion_df["prop_sector"] = round(participacion_df["Sector_indice"]/participacion_df["Conteo_Ciiu"],2)
    participacion_df["participacion"] = participacion_df.apply(lambda x: check_cliente(cliente=x['nombre_clean'], noticia=x['cuerpo_not_clean'], sector = x['prop_sector']), axis=1)    

    # Escritura del archivo de salida
    participacion_df['nombre_equipo'] = 'se cayo la app'
    participacion_df = participacion_df[['nombre_equipo','nit','news_id','participacion']]

    print("Escritura de archivo en output...")

    participacion_df.to_csv('src/data/output/categorizacion.csv',index=False)

    print('Se completo el proceso con exito!')
