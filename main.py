# Importamos las librerías necesarias.
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles  # Importar StaticFiles
from pydantic import BaseModel
import pandas as pd
import os
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Definir la ruta a la carpeta "dataset".
dataset_path = os.path.join(os.path.dirname(__file__), 'datasets')

# Crear un diccionario para almacenar los DataFrames.
dataframes = {}

# Listar los archivos en la carpeta "dataset" y cargar solo los CSV.
for filename in os.listdir(dataset_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(dataset_path, filename)
        try:
            dataframes[filename] = pd.read_csv(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al cargar {filename}: {str(e)}")

# Acceso a DataFrames
df_monthly = dataframes.get('PeliculasPorMesListo.csv')
df_daily = dataframes.get('PeliculasPorDiaListo.csv')
votes_df = dataframes.get('FuncionVotos.csv')
scores_df = dataframes.get('FuncionScore.csv')
resultado_crew = dataframes.get('resultado_crew.csv')
funcion_director = dataframes.get('FuncionDirector.csv')
resultado_cast_actores = dataframes.get('ResultadoCastActores.csv')
funcion_actor = dataframes.get('FuncionActor.csv')
lista_actores = dataframes.get('ListaActores.csv')
data = dataframes.get('data_reduction.csv')

# Chequeo de columnas requeridas
for df, required_columns in [
    (df_monthly, ['title', 'month']),
    (df_daily, ['title', 'day_of_week']),
    (votes_df, ['title', 'vote_count', 'vote_average']),
    (scores_df, ['title', 'release_year', 'popularity']),
    (resultado_crew, ['movie_id', 'name']),
    (funcion_director, ['id', 'revenue', 'return']),
    (resultado_cast_actores, ['movie_id', 'name']),
    (funcion_actor, ['id', 'return']),
    (lista_actores, ['name']),
]:
    if not all(column in df.columns for column in required_columns):
        raise HTTPException(status_code=500, detail="El DataFrame no contiene las columnas esperadas.")

# Diccionario de meses y días
meses_map = {
    'enero': 'January',
    'febrero': 'February',
    'marzo': 'March',
    'abril': 'April',
    'mayo': 'May',
    'junio': 'June',
    'julio': 'July',
    'agosto': 'August',
    'septiembre': 'September',
    'octubre': 'October',
    'noviembre': 'November',
    'diciembre': 'December'
}

dias_map = {
    'lunes': 'Monday',
    'martes': 'Tuesday',
    'miercoles': 'Wednesday',
    'jueves': 'Thursday',
    'viernes': 'Friday',
    'sabado': 'Saturday',
    'domingo': 'Sunday',
}

# Definición de clases para las respuestas
class MessageResponse(BaseModel):
    mensaje: str

class MovieInfo(BaseModel):
    title: str
    release_date: str
    return_: str
    budget: str
    revenue: str

class DirectorResponse(BaseModel):
    resultado_texto: str
    movies: List[MovieInfo]

# Normalizar nombres de columnas
df_monthly.columns = df_monthly.columns.str.lower()
df_daily.columns = df_daily.columns.str.lower()
votes_df.columns = votes_df.columns.str.lower()
scores_df.columns = scores_df.columns.str.lower()
resultado_crew.columns = resultado_crew.columns.str.lower()
funcion_director.columns = funcion_director.columns.str.lower()
resultado_cast_actores.columns = resultado_cast_actores.columns.str.lower()
funcion_actor.columns = funcion_actor.columns.str.lower()
lista_actores.columns = lista_actores.columns.str.lower()

# Inicializar el vectorizador y calcular la matriz TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['TokensLista'])

# Iniciar la API
app = FastAPI(
    title="API de Películas",
    description="Esta API permite consultar información sobre películas, sus votaciones, puntuaciones, directores y actores.",
    version="1.0.0",
)

# Montar la carpeta 'images' para servir archivos estáticos
app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    base_url = str(request.url).rstrip('/')
    return f"""
    <html>
        <head>
            <title>Bienvenido a la API de Películas</title>
            <style>
                body {{
                    background-image: url('/images/cinev2.jpg');
                    background-size: cover; /* Cubre todo el fondo */
                    background-position: center; /* Centra la imagen */
                    color: white; /* Cambia el color del texto para que sea más legible */
                }}
            </style>
        </head>
        <body>
            <h1>Bienvenido a la API de películas.</h1>
            <h2>Instrucciones:</h2>
            <p>Utiliza los siguientes endpoints para interactuar con la API:</p>
            <ul>
                <li><code>/peliculas/mes/?mes=nombre_del_mes</code>: cantidad de películas estrenadas en un mes (desde 1874 hasta 2020).</li>
                <li><code>/peliculas/dia/?dia=nombre_del_dia</code>: cantidad de películas estrenadas en un día (desde 1874 hasta 2020).</li>
                <li><code>/votes/?title=nombre_pelicula</code>: valoraciones de una película y valoración promedio.</li>
                <li><code>/score/?title=nombre_pelicula</code>: popularidad y año de estreno de una película.</li>
                <li><code>/titles/</code>: títulos disponibles para consultar.</li>
                <li><code>/actor/{{actor_name}}</code>: actuaciones de un actor, retorno, promedio.</li>
                <li><code>/actores</code>: actores disponibles para consultar.</li>
                <li><code>/director/{{director_name}}</code>: éxito de un director, ganancias de sus películas y más.</li>
                <li><code>/directores</code>: directores disponibles para consultar.</li>
                <li><code>/recommendations/?title={{tu_titulo}}</code>: recomendaciones de películas.</li>
            </ul>
            <h2>Links Ejemplo:</h2>
            <ul>
                <li><a href="{base_url}/peliculas/mes/?mes={list(meses_map.keys())[0]}">Para Mes: {list(meses_map.keys())[0]}</a></li>
                <li><a href="{base_url}/peliculas/dia/?dia={list(dias_map.keys())[0]}">Para Día: {list(dias_map.keys())[0]}</a></li>
                <li><a href="{base_url}/score/?title=Toy%20Story">Para Puntuación: Toy Story</a></li>
                <li><a href="{base_url}/votes/?title=Inception">Para Votación: Inception</a></li>
                <li><a href="{base_url}/titles/">Para Títulos</a></li>
                <li><a href="{base_url}/actor/Leonardo%20DiCaprio">Para Información del Actor: Leonardo DiCaprio</a></li>
                <li><a href="{base_url}/actores">Para Todos los Actores</a></li>
                <li><a href="{base_url}/director/Quentin%20Tarantino">Para Información del Director: Quentin Tarantino</a></li>
                <li><a href="{base_url}/directores">Para Todos los Directores</a></li>
                <li><a href="{base_url}/recommendations/?title=Inception">Para Recomendaciones</a></li>
            </ul>
        </body>
    </html>
    """

@app.get("/peliculas/mes/", response_class=HTMLResponse)
def cantidad_filmaciones_mes(mes: str):
    mes = mes.lower()              
    if mes not in meses_map:
        raise HTTPException(status_code=400, detail="Mes no válido. Por favor ingrese un mes en español.")

    mes_en_ingles = meses_map[mes]
    resultado = df_monthly[df_monthly['month'] == mes_en_ingles]
    cantidad = resultado['title'].count() if not resultado.empty else 0

    # Agregar HTML para la imagen
    return HTMLResponse(content=f"""
    <div>
        <p>La cantidad de películas que fueron estrenadas en el mes de <strong>{mes_en_ingles}</strong> es: <strong>{cantidad}</strong>.</p>
        <img src="/images/cinta3.jpg" alt="Cinta" style="width:100%; max-width:600px;"/>
    </div>
    """)


@app.get("/peliculas/dia/", response_class=HTMLResponse)
def cantidad_filmaciones_dia(dia: str):
    dia = dia.lower()
    if dia not in dias_map:
        raise HTTPException(status_code=400, detail="Día no válido. Por favor ingrese un día en español.")

    dia_en_ingles = dias_map[dia]
    cantidad = df_daily[df_daily['day_of_week'] == dia_en_ingles].shape[0]

    # Agregar HTML para la imagen
    return HTMLResponse(content=f"""
    <div>
        <p>La cantidad de películas que fueron estrenadas en el día <strong>{dia_en_ingles}</strong> es: <strong>{cantidad}</strong>.</p>
        <img src="/images/cinta2.jpg" alt="Cinta" style="width:100%; max-width:600px;"/>
    </div>
    """)

@app.get("/score/", response_class=HTMLResponse)
async def score_titulo(title: str):
    movie = scores_df[scores_df['title'].str.lower() == title.lower()]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")
    
    movie_data = movie.iloc[0]
    
    # Crear contenido HTML para la respuesta
    response_content = f"""
    <div>
        <p>La película '<strong>{movie_data['title']}</strong>' fue estrenada en el año <strong>{int(movie_data['release_year'])}</strong>, con una popularidad de <strong>{float(movie_data['popularity']):.2f}</strong>.</p>
        <img src="/images/pareja4.jpg" alt="Pareja" style="width:100%; max-width:600px;"/>
    </div>
    """
    
    return HTMLResponse(content=response_content)


@app.get("/titles/", response_class=HTMLResponse)
def get_titles():
    Titulos = votes_df['title'].unique().tolist()
    titulos_html = "<ul>"
    for titulo in Titulos:
        titulos_html += f"<li>{titulo}</li>"
    titulos_html += "</ul>"
    
    # HTML para la imagen fija a la derecha
    return HTMLResponse(content=f"""
    <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    color: black;
                }}
                .fixed-image {{
                    position: fixed;
                    top: 20px; /* Ajusta la posición vertical */
                    right: 20px; /* Ajusta la posición horizontal */
                    width: 450px; /* Ajusta el tamaño de la imagen */
                }}
            </style>
        </head>
        <body>
            <h2>Titulos disponibles:</h2>
            {titulos_html}
            <img src="/images/cinta4.jpg" alt="Cinta" class="fixed-image"/>
        </body>
    </html>
    """)


@app.get("/votes/", response_class=HTMLResponse)
async def votos_titulo(title: str):
    movie = votes_df[votes_df['title'].str.lower() == title.lower()]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")
    
    movie_data = movie.iloc[0]
    if movie_data['vote_count'] < 2000:
        return HTMLResponse(content=f"<p>La película '<strong>{movie_data['title']}</strong>' tuvo menos de 2000 valoraciones.</p>")
    else:
        response_content = f"""
        <p>La película '<strong>{movie_data['title']}</strong>' tuvo <strong>{int(movie_data['vote_count'])}</strong> votos y su puntaje promedio fue <strong>{float(movie_data['vote_average']):.2f}.</strong></p>
        <img src="/images/publico1.jpg" alt="Público" style="width:100%; max-width:600px;"/>
        """
        return HTMLResponse(content=response_content)

@app.get("/actor/{actor_name}", response_class=HTMLResponse)
def get_actor(actor_name: str):
    if not actor_name:
        raise HTTPException(status_code=400, detail="El nombre del actor no puede estar vacío.")
    
    actor_name_normalizado = actor_name.lower()
    peliculas_actor = resultado_cast_actores[resultado_cast_actores['name'].str.lower() == actor_name_normalizado]

    if peliculas_actor.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado")

    movie_ids = peliculas_actor['movie_id'].tolist()
    ganancias_actor = funcion_actor[funcion_actor['id'].isin(movie_ids)]

    retorno_total = ganancias_actor['return'].sum()
    ganancias_validas = ganancias_actor[ganancias_actor['return'] > 0]
    cantidad_peliculas_validas = len(ganancias_validas)

    if cantidad_peliculas_validas > 0:
        retorno_promedio = round(ganancias_validas['return'].mean(), 2) * 100
    else:
        retorno_promedio = 0.0

    retorno_total_formateado = f"{retorno_total * 100:,.2f}%"
    retorno_promedio_formateado = f"{retorno_promedio:,.2f}%"

    peliculas_con_return_zero = ganancias_actor[ganancias_actor['return'] == 0]['id'].tolist()
    peliculas_con_return_zero_count = len(peliculas_con_return_zero)

    resultado_texto = (
        f"El actor <strong>{actor_name}</strong> ha actuado en <strong>{len(ganancias_actor)}</strong> películas, "
        f"con un retorno total de <strong>{retorno_total_formateado}</strong>, "
        f"y un retorno promedio de <strong>{retorno_promedio_formateado}</strong>. "
        f"La cantidad de películas sin retorno en el dataset son <strong>{peliculas_con_return_zero_count}</strong>, "
        f"el retorno promedio contándolas es de <strong>{round(retorno_total / len(ganancias_actor) * 100, 2):,.2f}%</strong>."
    )

    # Incluir HTML para la imagen
    return HTMLResponse(content=f"""
    <div style="display: flex; justify-content: space-between; max-width: 80%; margin: auto;">
        <div style="flex: 1; max-width: 50%;">
            <p>{resultado_texto}</p>
        </div>
        <img src="/images/oscar2.jpg" alt="Oscar" style="width: 60%; max-width: 500px;"/>
    </div>
    """)

@app.get("/actores", response_class=HTMLResponse)
def listar_actores():
    actores_lista = lista_actores['name'].tolist()  # Obtener lista de actores
    actores_html = "<ul>"
    for actor in actores_lista:
        actores_html += f"<li>{actor}</li>"
    actores_html += "</ul>"
    
    # HTML con la imagen fija a la derecha
    return HTMLResponse(content=f"""
    <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    color: black;
                }}
                .fixed-image {{
                    position: fixed;
                    top: 20px; /* Ajusta la posición vertical */
                    right: 20px; /* Ajusta la posición horizontal */
                    width: 800px; /* Ajusta el tamaño de la imagen */
                }}
            </style>
        </head>
        <body>
            <p>Actores disponibles:</p>
            {actores_html}
            <img src="/images/alfombraroja.jpg" alt="Alfombra Roja" class="fixed-image"/>
        </body>
    </html>
    """)

@app.get("/director/{director_name}", response_class=HTMLResponse)
def get_director(director_name: str):
    director_name_lower = director_name.lower()
    director_movies = resultado_crew[resultado_crew['name'].str.lower() == director_name_lower]

    if director_movies.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado")

    director_movies = director_movies.merge(funcion_director, left_on='movie_id', right_on='id', how='inner')

    total_revenue = director_movies['revenue'].sum()
    total_return = director_movies['return'].sum()
    average_return = total_return / len(director_movies) if len(director_movies) > 0 else 0

    non_zero_returns = director_movies[director_movies['return'] > 0]
    average_return_non_zero = non_zero_returns['return'].mean() if len(non_zero_returns) > 0 else 0

    total_movies = len(director_movies)
    zero_return_movies = director_movies[director_movies['return'] == 0]
    total_zero_return = len(zero_return_movies)

    movies_info = [
        {
            "title": row['title'],
            "release_date": row['release_date'],
            "return": f"{row['return']:.2f}%",
            "budget": f"${row['budget']:,.2f}",
            "revenue": f"${row['revenue']:,.2f}"
        } for index, row in director_movies.iterrows()
    ]

    # Construir el texto del resultado
    resultado_texto = (
        f"El director <strong>{director_name}</strong> ha obtenido una ganancia total de <strong>{total_revenue:,.2f}</strong>, "
        f"con un retorno total promedio de <strong>{average_return:.2f}%</strong> en un total de <strong>{total_movies}</strong> películas, "
        f"y con un retorno de <strong>{average_return_non_zero:.2f}%</strong> sin contar las "
        f"películas que no tienen retorno en este dataset.({total_zero_return})"
    )

    
        # Generar la lista de películas
    movies_list = "<ul>"
    for movie in movies_info:
        movies_list += (
            f"<li><strong>{movie['title']}</strong> - <br>"
            f"Fecha de lanzamiento: {movie['release_date']}, <br>"
            f"Retorno: {movie['return']}, <br>"
            f"Costo: {movie['budget']}, <br>"
            f"Ganancia: {movie['revenue']}</li><br>"
        )
    movies_list += "</ul>"

    # Combinar el texto con la lista de películas
    full_content = f"<p>{resultado_texto}</p><h3>Películas dirigidas:</h3>{movies_list}"

    # Agregar HTML para la imagen a la derecha
    return HTMLResponse(content=f"""
    <div style="display: flex; justify-content: space-between; max-width: 85%; margin: auto;">
        <div style="flex: 1; max-width: 60%;">
            {full_content}
        </div>
        <img src="/images/popcorn.jpg" alt="Palomitas" style="width: 550px; height: 700px; margin-top: 40px;"/> 
</div>
""")


@app.get("/directores", response_class=HTMLResponse)
def obtener_directores():
    directores = resultado_crew['name'].unique().tolist()
    directores_html = "<ul>"
    for director in directores:
        directores_html += f"<li>{director}</li>"
    directores_html += "</ul>"
    
    # HTML para la imagen fija a la derecha
    return HTMLResponse(content=f"""
    <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    color: black;
                }}
                .fixed-image {{
                    position: fixed;
                    top: 20px; /* Ajusta la posición vertical */
                    right: 20px; /* Ajusta la posición horizontal */
                    width: 600px; /* Ajusta el tamaño de la imagen */
                }}
            </style>
        </head>
        <body>
            <p>Directores disponibles:</p>
            {directores_html}
            <img src="/images/director.jpg" alt="Director" class="fixed-image"/>
        </body>
    </html>
    """)


@app.get("/recommendations/", response_class=HTMLResponse)
def recomendacion(title: str):
    titulo_ingresado = title.lower()

    data['lower_title'] = data['title'].str.lower()

    if titulo_ingresado not in data['lower_title'].values:
        raise HTTPException(status_code=404, detail="Movie not found")

    idx = data[data['lower_title'] == titulo_ingresado].index[0]

    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    recommendations_indices = cosine_similarities.argsort()[-6:-1][::-1]

    recommendations = data.iloc[recommendations_indices]
    recommendations['similarity'] = cosine_similarities[recommendations_indices]
    recommendations = recommendations.sort_values(by='vote_average', ascending=False)

    recommendations_html = "<ul>"
    for _, row in recommendations.iterrows():
        recommendations_html += f"<li><strong>{row['title']}</strong> - Voto promedio: <strong>{row['vote_average']}</strong>, Similitud: <strong>{row['similarity']:.4f}</strong></li>"
    recommendations_html += "</ul>"

    # Combinar el texto de recomendaciones con la imagen
    return HTMLResponse(content=f"""
    <div style="display: flex; justify-content: space-between; max-width: 80%; margin: auto;">
        <div style="flex: 1; max-width: 50%;">
            <p>Recomendaciones basadas en '<strong>{title}</strong>': {recommendations_html}</p>
        </div>
        <img src="/images/publico.jpg" alt="Público" style="width: 600px; height: auto; margin-left: 20px;"/>
    </div>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
