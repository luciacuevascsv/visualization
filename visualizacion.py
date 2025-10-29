from dash import Dash, dcc, html, Input, Output, State
from graficas import grafica_rank, media_line_region, media_bar_region, paises_graph, media_graph, mapa_graph, radar_factores_region, happiness_forecast
import pandas as pd
from dash import ctx

data = pd.read_csv("happiness.csv")

app = Dash(__name__)

app.layout = html.Div([

    # TODO : COLUMNAS

    # COLUMNA IZQUIERDA: COLUMNAS
    html.Div([

        # TEXTO TITULO 
        html.H2("Índice de Felicidad", style={'justify-content': 'center', 'text-align': 'center', 'margin': '20px auto 10px auto'}),

        # FILTRO AÑOS
        html.Div([
            html.Label("Rango de años:", style={}),
            dcc.RangeSlider(
                id='year-slider',
                min=int(data['year'].min()),
                max=int(data['year'].max()),
                step=1,
                value=[int(data['year'].min()), int(data['year'].max())],
                marks={i: str(i) for i in range(int(data['year'].min()), int(data['year'].max())+1)},
                vertical=True,
                tooltip={"placement": "bottom"},
            )
        ], style={'justify-content': 'center','margin': '10px auto auto auto', 'text-align': 'center'})

    ], style={

        'display': 'flex',
        'flex-direction': 'column',
        'background-color': 'white',
        'gap':'20px',
        'justify-content': 'center',
        'text-align': 'center',
        'margin': 'auto',
        'borderRadius': '5px',
        'position': 'fixed',   
        'top': '20px',        
        'left': '20px',        
        'width': '150px',      
        'zIndex': '1000',      
        'overflow': 'auto',
        'height':'100%'    
    }),

    # CONTENIDO: FILAS
    html.Div([

        # PRIMERA FILA: COLUMNAS
        html.Div([

            # FILTRO AÑO
            html.Div([
                html.Label("Año:", style={"flex": "0.05"}),
                html.Div([
                    dcc.Slider(
                        id='year',
                        min=int(data['year'].min()),  
                        max=int(data['year'].max()), 
                        step=1,
                        value=int(data['year'].max()),  
                        marks={i: str(i) for i in range(int(data['year'].min()), int(data['year'].max())+1)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={"flex": "0.95"}),
            ], style={'width':'70%', 'margin':'auto', 'text-align': 'center', 'justify-content': 'center', }),

            # PARTE ARRIBA: FILAS
            html.Div([

                # PLOT MAPA -- SELECCIONABLE
                html.Div([

                    html.Div([
                        html.P("SELECCIONABLE", style={'fontSize': '10px', 'color': 'lightgrey', 'fontWeight': 'bold'}),
                        html.Button("Deseleccionar país", id='boton-deseleccionar-pais', n_clicks=0, style={'display': 'none'}),
                    ], style={'display':'flex', 'flex-direction': 'row', 'gap':'10px', 'justify-content': 'center', 'text-align': 'center', 'margin': 'auto', 'alignItems': 'center'}),

                    html.P(id='mapa-titulo', style={'width': '100%', 'margin': 'auto'}),

                    dcc.Graph(id='graf1', style={'margin': 'auto', 'justify-content': 'center', 'text-align': 'center', 'width': '100%'})

                ], style={'border': '4px dashed rgba(211, 211, 211, 0.5)', 'borderRadius': '5px', 'padding': '0px 0px 10px 0px', 'justify-content': 'center', 'text-align': 'center'}),

                # PARTE DERECHA: FILAS
                html.Div([

                    # PARTE ARRIBA: FILAS
                    html.Div([

                        # PLOT MEDIA
                        html.Div([
                            html.P(id='graficas-mapa-titulo_izq', style={'width': '100%', 'margin': 'auto auto 5px auto'}),
                            dcc.Graph(id='graf2', style={}),
                            html.P(id='graficas-mapa-leyenda-izq', style={'width': '70%', 'margin': 'auto auto 5px auto'})
                        ], style={'margin': 'auto', 'flex':'0.5', 'justify-content': 'center', 'text-align': 'center', 'display': 'flex', 'flex-direction': 'column'}),

                        # PLOT PAISES
                        html.Div([
                            html.P(id='graficas-mapa-titulo_der', style={'width': '100%', 'margin': 'auto auto 5px auto'}),
                            dcc.Graph(id='graf3', style={}),
                            html.P(id='graficas-mapa-leyenda-der', style={'width': '70%', 'margin': 'auto auto 5px auto'})
                        ], style={'margin': 'auto', 'flex':'0.5', 'justify-content': 'center', 'text-align': 'center', 'display': 'flex', 'flex-direction': 'column'})

                    ], style={'display': 'flex', 'flex-direction': 'row', 'gap':'10px'}),


                    # PARTE ABAJO: FILAS
                    html.Div([

                        # PLOT PREDICCIÓN
                        dcc.Graph(id='graf8', style={'justify-content': 'center', 'text-align': 'center', 'margin': 'auto', "flex": "0.8"}),

                        # FILTRO NUMERO AÑOS
                        html.Div([
                            html.Label("Años:", style={}),
                            dcc.Input(
                                id='num-años',
                                type='number',
                                min=1,
                                max=5,
                                step=1,
                                value=2,
                                style={'width': '30px'}
                            )

                        ], style={'display': 'flex', 'flex-direction': 'column', 'gap': '10px',  'justify-content': 'center', 'borderRadius': '5px', 'text-align': 'center', 'flex':'0.4', 'background-color': '#f7f7f8',  'padding': '10px', 'borderRadius': '5px', 'width':'100%'}),


                    ], style={'display': 'flex', 'flex-direction': 'row', 'gap':'10px','justify-content': 'center', 'text-align': 'center', 'margin':'auto'})

                ], style={'display': 'flex', 'flex-direction': 'column', 'gap':'10px'})

            ], style={'display': 'flex', 'flex-direction': 'row', 'gap':'10px', 'width': '100%', 'margin':'auto', 'text-align': 'center', 'justify-content': 'center'}),

            # FILTRO GROUP BY
            html.Div([
                html.Label("Agrupar por:", style={'justify-content': 'center', 'text-align': 'center', 'margin':'auto 10px auto auto'}),
                html.Div([
                    dcc.Dropdown(id='group_by_barras', options=[{'label': 'Region', 'value': False}, {'label': 'Continente', 'value': True}], value=True, clearable=True)
                ], style={'width':'30%', 'justify-content': 'center', 'text-align': 'center', 'margin':'auto auto auto 10px'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'gap': '10px',  'justify-content': 'center', 'borderRadius': '5px', 'text-align': 'center', 'background-color': '#f7f7f8',  'padding': '10px', 'borderRadius': '5px', 'width':'70%', 'margin':'auto'}),



            # PARTE ABAJO: FILAS
            html.Div([

                # PLOT BARRAS -- SELECCIONABLE
                html.Div([
                    html.Div([
                        html.P("SELECCIONABLE", style={'fontSize': '10px', 'color': 'lightgrey', 'fontWeight': 'bold'}),
                        html.Button("Deseleccionar región", id='boton-deseleccionar-region', n_clicks=0, style={'display': 'none'}),
                    ], style={'margin': '-10px auto 10px auto', 'display':'flex', 'flex-direction': 'row', 'gap':'10px', 'justify-content': 'center', 'text-align': 'center', 'margin': 'auto'}),

                    dcc.Graph(id='graf4', style={'justify-content': 'center', 'text-align': 'center', 'margin': 'auto'})

                ], style={'display': 'flex', 'flex-direction': 'column', 'gap':'10px', 'border': '4px dashed rgba(211, 211, 211, 0.5)', 'borderRadius': '5px'}),

                
                # PLOT MEDIAS REGIONES
                html.Div([
                    dcc.Graph(id='graf5', style={}),
                    html.P(id='graficas_media_leyenda')
                ], style={'display': 'flex', 'flex-direction': 'column', 'gap': '10px', 'justify-content': 'center', 'text-align': 'center'})

            ], style={'display': 'flex', 'flex-direction': 'row', 'gap':'10px', 'width': '100%', 'margin':'auto', 'text-align': 'center', 'justify-content': 'center'})

        ], style={'display': 'flex', 'flex-direction': 'column',  'background-color': 'white', 'gap':'30px', 'borderRadius': '5px', 'padding': '25px', 'margin': '10px 0px'}),

        # SEGUNDA FILA: FILAS
        html.Div([

            # PARTE IZQUIERDA: COLUMNAS
            html.Div([

                # FILTRO TOP PAISES
                html.Div([
                    html.Label("Países:", style={'text-align': 'center', 'justify-content': 'center', 'margin': 'auto auto 10px auto'}),
                    html.Div([
                        dcc.Slider(
                            id='top-paises',
                            min=5,  
                            max=15, 
                            step=1,
                            value=9,  
                            marks={i: str(i) for i in range(5,16)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width':'100%'})
                ], style={'display': 'flex', 'flex-direction': 'column'}),

                dcc.Graph(id='graf7', style={'justify-content': 'center', 'text-align': 'center', 'margin': 'auto'})

            ], style={'display': 'flex', 'flex-direction': 'column', 'gap':'10px', 'background-color': 'white', 'borderRadius': '5px', 'padding': '25px'}),

            # PARTE DERECHA: COLUMNAS
            html.Div([

                # PARTE ARRIBA: FILAS
                html.Div([

                    # FILTRO AGRUPAR POR
                    html.Div([
                        html.Label("Agrupar por:", style={'text-align': 'center', 'justify-content': 'center', 'margin': 'auto 10px auto auto'}),
                        html.Div([
                            dcc.Dropdown(id='group_by_factores', options=[{'label': 'Region', 'value': False}, {'label': 'Continente', 'value': True}], value=True, clearable=True)
                        ], style={'width':'60%'})
                    ], style={'flex':'0.6','display': 'flex', 'flex-direction': 'row', 'gap': '10px',  'justify-content': 'center', 'text-align': 'center', 'margin': 'auto', 'background-color': '#f7f7f8',  'padding': '10px', 'borderRadius': '5px'}),

                    # FILTRO NORMALIZAR
                    html.Div([
                        html.Label("Normalizar:", style={}),
                        dcc.Checklist(
                                id='normalizar_factores',
                                options=[{'label': '', 'value': False}],
                                value=[]
                        )

                    ], style={'flex':'0.2', 'display': 'flex', 'flex-direction': 'row', 'gap': '10px',  'justify-content': 'center', 'text-align': 'center', 'margin': 'auto', 'flex':'0.4', 'background-color': '#f7f7f8',  'padding': '10px', 'borderRadius': '5px'})

                ], style={'display': 'flex', 'flex-direction': 'row', 'gap':'10px'}),

                dcc.Graph(id='graf6', style={'justify-content': 'center', 'text-align': 'center', 'margin': 'auto'})

            ], style={'display': 'flex', 'flex-direction': 'column', 'gap':'10px', 'background-color': 'white', 'borderRadius': '5px', 'padding': '25px'})

        ], style={'display': 'flex', 'flex-direction': 'row', 'gap':'25px', 'width': '70%', 'margin':'auto', 'text-align': 'center', 'justify-content': 'center'})

    ], style={'flex':'0.9', 'display': 'flex', 'flex-direction': 'column', 'gap':'20px', 'margin-left': '180px'}),

    dcc.Store(id='pais-seleccionado', storage_type='memory')

], style={'display': 'flex', 'flex-direction': 'row', 'background-color': '#f7f7f8', 'fontFamily': 'Segoe UI', 'gap':'20px'})




@app.callback(
    Output('graf1', 'figure'),
    Output('graf2', 'figure'),
    Output('graf3', 'figure'),
    Output('graf4', 'figure'),
    Output('graf5', 'figure'),
    Output('graf6', 'figure'),
    Output('graf7', 'figure'),
    Output('graf8', 'figure'),
    Input('year-slider', 'value'),
    Input('year', 'value'),
    Input('group_by_barras', 'value'),
    Input('group_by_factores', 'value'),
    Input('top-paises', 'value'),
    Input('num-años', 'value'),
    Input('graf4', 'clickData'),
    Input('pais-seleccionado', 'data'),
    Input('normalizar_factores', 'value'),
    Input('boton-deseleccionar-pais', 'n_clicks'),
    Input('boton-deseleccionar-region', 'n_clicks'),
    prevent_initial_call=True

)
def update_graphs(years, year, group_by_barras, group_by_factores, top_paises, num_años, click_region, pais_seleccionado, normalizar_factores, n_clicks_mapa, n_clicks_region):
    if ctx.triggered_id == 'boton-deseleccionar-pais':
        pais_seleccionado = None

    if ctx.triggered_id == 'boton-deseleccionar-region':
        click_region = None  
    region_media_bar_region = None
    if click_region and 'points' in click_region:
        region_media_bar_region = click_region['points'][0]['x'].replace("<br>", " ")

    fig1 = mapa_graph(data, year)
    fig2=media_graph(data, years, year, country=pais_seleccionado)
    fig3=paises_graph(data, years, year, country=pais_seleccionado)
    fig4=media_bar_region(data, year, group_by_barras)
    fig5=media_line_region(data, years, year, region_media_bar_region, group_by_barras)
    fig6=radar_factores_region(data, years, group_by_factores, normalizar_factores)
    fig7=grafica_rank(data, top_paises, years)
    fig8=happiness_forecast(data, years, year, pais_seleccionado, num_años)

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8

@app.callback(
    Output('year', 'min'),
    Output('year', 'max'),
    Output('year', 'value'),
    Input('year-slider', 'value')
)
def update_map_year_range(year_range):
    start, end = year_range
    return start, end, end  

@app.callback(
    Output('mapa-titulo', 'children'),
    Input('year', 'value')
)
def actualizar_texto(año):
    return f"Mapa de Índice de Felicidad en {año}"

@app.callback(
    Output('graficas-mapa-titulo_izq', 'children'),
    Output('graficas-mapa-titulo_der', 'children'),
    Output('graficas-mapa-leyenda-izq', 'children'),
    Output('graficas-mapa-leyenda-der', 'children'),
    Output('boton-deseleccionar-pais', 'style'),
    Input('pais-seleccionado', 'data'),
    Input('boton-deseleccionar-pais', 'n_clicks'),

)
def actualizar_texto_mapa(country_mapa_graph, n_clicks_mapa):
    if country_mapa_graph:
        texto_izq = f"Evolución de {country_mapa_graph} respecto a la media"
        texto_der = f"{country_mapa_graph} contra la media"
        texto_leyenda_izq = html.Div([
            html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px', 
                             'backgroundColor': '#f79343', 'borderRadius': '50%', 'marginRight': '5px'}),
            f"Media de {country_mapa_graph}",
            html.Br(),
            html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px', 
                             'backgroundColor': '#7e4d8f', 'borderRadius': '50%', 'marginRight': '5px'}),
            "Media global (MG)"
        ])

        texto_leyenda_der = html.Div([
            html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px', 
                             'backgroundColor': '#f79343', 'borderRadius': '50%', 'marginRight': '5px'}),
            f"Media de {country_mapa_graph} >= MG",
            html.Br(),
            html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px', 
                             'backgroundColor': '#7e4d8f', 'borderRadius': '50%', 'marginRight': '5px'}),
            f"Media {country_mapa_graph} < MG"
        ])

        style_boton = {'margin-left': '5px'}
    else:
        texto_izq = f"Evolución de la media"
        texto_der = f"Países contra la media"
        texto_leyenda_izq = html.Div([
            html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px', 
                             'backgroundColor': '#7e4d8f', 'borderRadius': '50%', 'marginRight': '5px'}),
            "Media global (MG)"
        ])

        texto_leyenda_der = html.Div([
            html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px', 
                             'backgroundColor': '#f79343', 'borderRadius': '50%', 'marginRight': '5px'}),
            f"Países con media >= MG",
            html.Br(),
            html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px', 
                             'backgroundColor': '#7e4d8f', 'borderRadius': '50%', 'marginRight': '5px'}),
            f"Países con media < MG"
        ])

        style_boton = {'display': 'none'}

    return texto_izq, texto_der, texto_leyenda_izq, texto_leyenda_der, style_boton

@app.callback(
    Output('boton-deseleccionar-region', 'style'),
    Output('graficas_media_leyenda', 'children'),
    Input('graf4', 'clickData'),
    Input('boton-deseleccionar-region', 'n_clicks')  
)
def actualizar_texto_region(click_region, n_clicks_region):
    if ctx.triggered_id == 'boton-deseleccionar-region':
        click_region = None  

    region_media_bar_region = None
    if click_region and 'points' in click_region:
        region_media_bar_region = click_region['points'][0]['x'].replace("<br>", " ")

    if click_region and 'points' in click_region:
        texto_leyenda= html.Div([
            html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px', 
                             'backgroundColor': '#f79343', 'borderRadius': '50%', 'marginRight': '5px'}),
            f"Evolución de la media de {region_media_bar_region}",
            html.Br(),
            html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px', 
                             'backgroundColor': '#7e4d8f', 'borderRadius': '50%', 'marginRight': '5px'}),
            f"Evolución de la media global (MG)"
        ])
        style_boton = {'margin-left': '5px'}
    else:
        texto_leyenda = html.Div([
            html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px', 
                             'backgroundColor': '#7e4d8f', 'borderRadius': '50%', 'marginRight': '5px'}),
            "Evolución de la media global (MG)"
        ])
        style_boton = {'display': 'none'}
    
    return  style_boton, texto_leyenda

@app.callback(
    Output('pais-seleccionado', 'data'),
    Input('graf1', 'clickData'),
    Input('boton-deseleccionar-pais', 'n_clicks'),
    State('pais-seleccionado', 'data'),
    prevent_initial_call=True
)
def actualizar_pais(click_mapa, pais_actual, n_clicks_mapa):
    ctx_trigger = ctx.triggered_id

    if ctx_trigger == 'boton-deseleccionar-pais':
        return None

    if ctx_trigger == 'graf1' and click_mapa and 'points' in click_mapa:
        return click_mapa['points'][0]['location']

    return pais_actual

if __name__ == '__main__':
    app.run(debug=True)

