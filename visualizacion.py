from dash import Dash, dcc, html, Input, Output
from graficas import grafica_rank, score_by_region, peso_por_factor, scatter_happiness, happiness_map
import pandas as pd

data = pd.read_csv("happiness.csv")

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H2(
            "A Decade of Happiness: Global Rankings and Factors",
            style={'text-align': 'center', 'margin-bottom': '10px'}
        ),

        html.Div([
            html.Div([
                html.Label("Year range:", style={'display':'flex','text-align': 'center', 'justify-content': 'center'}),
                dcc.RangeSlider(
                    id='year-slider',
                    min=int(data['year'].min()),
                    max=int(data['year'].max()),
                    step=1,
                    value=[int(data['year'].min()), int(data['year'].max())],
                    marks={i: str(i) for i in range(int(data['year'].min()), int(data['year'].max())+1)}
                )
            ], style={'flex': 0.7, 'background-color': "#f7f7f8", 'border-radius': '20px', 'padding': '10px 20px', 'box-shadow': '0 4px 12px rgba(0,0,0,0.05)', 'margin-right':'10px'}),

            html.Div([
                html.Div([
                    html.Label("Group by:", style={
                        'margin-right': '10px',
                        'font-weight': '600',
                        'font-size': '16px',
                        'align-self': 'center'
                    }),
                    dcc.RadioItems(
                        id='group-radio',
                        options=[
                            {'label': 'Region', 'value': False},
                            {'label': 'Continent', 'value': True}
                        ],
                        value=True,
                        inline=True,
                        style={'align-self': 'center'}
                    )
                ], style={
                    'display': 'flex',
                    'justify-content': 'center',
                    'align-items': 'center',
                    'gap': '10px',
                    'margin-bottom': '10px'
                }),
                html.P(
                    "Note: This filter does not apply to the first and last graphs.",
                    style={
                        'text-align': 'center',
                        'color': '#555',
                        'font-size': '14px',
                        'margin': '0'
                    }
                )
            ], style={
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'center',
                'justify-content': 'center',
                'flex': 0.3,
                'background-color': '#f7f7f8',
                'border-radius': '20px',
                'padding': '20px 20px',
                'box-shadow': '0 4px 12px rgba(0,0,0,0.05)',
                'margin-left': '10px'
            })
        ], style={
            'width': '70%',
            'margin': '10px auto 30px auto',
            'display': 'flex',
            'align-items': 'center'
        }),
    ],
    style={
        'position': 'sticky',    
        'top': '0',             
        'zIndex': 1000,          
        'background-color': 'white', 
        'padding': '10px 0',
        'box-shadow': '0 2px 8px rgba(0,0,0,0.05)'
    }),

    html.Div([
        html.Div([
            html.Div([
                html.Label("Top N countries: ", style={'margin-right': '10px'}),
                dcc.Input(id='top-n', type='number', value=7, min=1, step=1, style={'width': '20%'})
                ], style={'display':'flex','text-align': 'center', 'justify-content': 'center', 'background-color': "#ffffff", 'border-radius': '20px', 'box-shadow': '0 4px 12px rgba(0,0,0,0.05)', 'padding': '5px'}),
            html.P("Non–Western European countries are outlined by region.", style={'text-align': 'center', 'color': '#555', 'font-size': '16px', 'margin-left': '50px'})
            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px', 'justify-content': 'center'}),
        dcc.Graph(id='graf1')
    ], style={'width': '70%', 'margin': 'auto', 'background-color': "#f7f7f8", 'border-radius': '20px', 'box-shadow': '0 4px 12px rgba(0,0,0,0.05)', 'padding': '20px'}),

    html.Div([
        html.Div([
            html.Div([
                html.Label("Ranges:"),
                dcc.RadioItems(
                    id='violin',
                    options=[
                        {'label': 'Violin', 'value': True},
                        {'label': 'Line', 'value': False}
                    ],
                    value=True,
                    inline=True
                )
            ], style={'display': 'flex', 'align-items': 'center', 'background-color': "#ffffff", 'border-radius': '20px', 'box-shadow': '0 4px 12px rgba(0,0,0,0.05)', 'padding': '10px 20px',  'margin-right':'25px'}),
            html.Div([
                html.Label("Means:"),
                dcc.RadioItems(
                    id='means',
                    options=[
                        {'label': 'Show', 'value': True},
                        {'label': 'Hide', 'value': False}
                    ],
                    value=True,
                    inline=True
                )
            ], style={'display': 'flex', 'align-items': 'center', 'margin-left':'25px', 'background-color': "#ffffff", 'border-radius': '20px', 'box-shadow': '0 4px 12px rgba(0,0,0,0.05)', 'padding': '10px 20px'}),
            html.P("Vertical lines/violins represent the ranges (min-max) of Happiness Score by region. ", style={'text-align': 'center', 'color': '#555', 'font-size': '16px',  'margin-left':'25px'})

        ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px', 'justify-content': 'center'}),
        dcc.Graph(id='graf2')
    ], style={'width': '70%', 'margin': '20px auto', 'background-color': "#f7f7f8", 'border-radius': '20px', 'box-shadow': '0 4px 12px rgba(0,0,0,0.05)', 'padding': '20px'}),

    html.Div([
        html.P("The factor contribution explains how much each factor adds to the region's average Happiness Score.", style={'text-align': 'center', 'color': '#555', 'font-size': '16px', 'margin-bottom': '20px'}),
        dcc.Graph(id='graf3')
    ], style={'width': '70%', 'margin': '20px auto', 'background-color': "#f7f7f8", 'border-radius': '20px', 'box-shadow': '0 4px 12px rgba(0,0,0,0.05)', 'padding': '20px'}),

    html.Div([
        html.Div([
            html.Div([
                html.Label("Factor:", style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='factor',
                    options=[
                        {'label': 'GDP per Capita', 'value': 'GDP per Capita'},
                        {'label': 'Social Support', 'value': 'Social Support'},
                        {'label': 'Healthy Life Expectancy', 'value': 'Healthy Life Expectancy'},
                        {'label': 'Freedom', 'value': 'Freedom'},
                        {'label': 'Perceptions of Corruption', 'value': 'Perceptions of Corruption'},
                        {'label': 'Generosity', 'value': 'Generosity'}
                    ],
                    value='GDP per Capita',
                    clearable=False,
                    style={'width': '100%'}
                )
            ],
            style={
                'background-color': "#ffffff",
                'border-radius': '20px',
                'box-shadow': '0 4px 12px rgba(0,0,0,0.05)',
                'padding': '10px 20px',
                'width': '20%',
                'display': 'flex', 'align-items': 'center', 'justify_content':'center'
            }),
            html.Div([
                html.Label("Trend:"),
                dcc.RadioItems(
                    id='trend',
                    options=[
                        {'label': 'Show', 'value': True},
                        {'label': 'Hide', 'value': False}
                    ],
                    value=True,
                    inline=True
                )
            ], style={'display': 'flex', 'align-items': 'center', 'background-color': "#ffffff", 'border-radius': '20px', 'box-shadow': '0 4px 12px rgba(0,0,0,0.05)', 'padding': '10px 20px', 'margin': '0px 50px'}),
            html.Div([
                html.Label("Points:"),
                dcc.RadioItems(
                    id='points',
                    options=[
                        {'label': 'Show', 'value': True},
                        {'label': 'Hide', 'value': False}
                    ],
                    value=False,
                    inline=True
                )
            ], style={'display': 'flex', 'align-items': 'center', 'background-color': "#ffffff", 'border-radius': '20px', 'box-shadow': '0 4px 12px rgba(0,0,0,0.05)', 'padding': '10px 20px'})
        ], style={'display': 'flex', 'gap': '10px', 'margin-bottom': '10px', 'align-items': 'center', 'justify-content': 'center', 'padding': 'auto 50px'}),
        dcc.Graph(id='graf4')
    ], style={'width': '70%', 'margin': '20px auto', 'background-color': "#f7f7f8", 'border-radius': '20px', 'box-shadow': '0 4px 12px rgba(0,0,0,0.05)', 'padding': '20px'}),

    html.Div([
        html.Div([
            html.Div([
                html.Label("Map Year:", style={'margin-right': '10px'}),
                dcc.Input(
                    id='map-year',
                    type='number',
                    value=int(data['year'].max()),
                    min=int(data['year'].min()),
                    max=int(data['year'].max()),
                    step=1,
                    style={'width': '80%'}
                )
            ], style={
                'background-color': "#ffffff",
                'border-radius': '20px',
                'box-shadow': '0 4px 12px rgba(0,0,0,0.05)',
                'padding': '10px 20px',
                'width': '10%',
                'display': 'flex', 'align-items': 'center', 'justify_content':'center'
            }),
            html.Div([
                html.Label("Map Projection:", style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='projection-dropdown',
                    options=[
                        {'label': 'Equirectangular', 'value': 'equirectangular'},
                        {'label': 'Mercator', 'value': 'mercator'},
                        {'label': 'Orthographic', 'value': 'orthographic'},
                        {'label': 'Natural Earth', 'value': 'natural earth'}
                    ],
                    value='natural earth',
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style={
                'background-color': "#ffffff",
                'border-radius': '20px',
                'box-shadow': '0 4px 12px rgba(0,0,0,0.05)',
                'padding': '10px 20px',
                'width': '25%',
                'display': 'flex', 'align-items': 'center', 'justify_content':'center'
            })
        ], style={
            'display': 'flex',
            'gap': '20px',
            'justify-content': 'center',
            'margin-bottom': '15px'
        }),

        dcc.Graph(id='graf5', style={'height': '500px'})

    ], style={
        'width': '70%',
        'margin': '20px auto',
        'background-color': "#f7f7f8",
        'border-radius': '20px',
        'box-shadow': '0 4px 12px rgba(0,0,0,0.05)',
        'padding': '20px'
    })
])


@app.callback(
    Output('graf1', 'figure'),
    Output('graf2', 'figure'),
    Output('graf3', 'figure'),
    Output('graf4', 'figure'),
    Output('graf5', 'figure'),
    Input('year-slider', 'value'),
    Input('group-radio', 'value'),
    Input('top-n', 'value'),
    Input('map-year', 'value'),
    Input('projection-dropdown', 'value'),
    Input('violin', 'value'),
    Input('means', 'value'),
    Input('factor', 'value'),
    Input('trend', 'value'),
    Input('points', 'value'),
)
def update_graphs(years, group_by_region_agregada, top_n, map_year, projection_type, violin, show_means, factor, show_trend, show_points):
    fig1 = grafica_rank(data, top_n=top_n, years=years)
    fig2 = score_by_region(data, violin=violin, show_means=show_means, group_by_region_agregada=group_by_region_agregada, years=years)
    fig3 = peso_por_factor(data, group_by_region_agregada=group_by_region_agregada, years=years)
    fig4 = scatter_happiness(data, group_by_region_agregada=group_by_region_agregada, years=years, factor=factor, show_trend=show_trend, show_points=show_points)
    fig5 = happiness_map(data, year=map_year, projection_type=projection_type)
    return fig1, fig2, fig3, fig4, fig5

if __name__ == "__main__":
    app.run(debug=True)
