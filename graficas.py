import pandas as pd           
import numpy as np          
from collections import Counter  

import plotly.graph_objects as go  
import plotly.express as px        

from plotly.express import colors
from sklearn.linear_model import LinearRegression



################


def mapa_graph(data, year):
    df_year = data[data['year'] == year].copy()

    fig = px.choropleth(
        df_year,
        locations="Country",
        locationmode="country names",
        color="Happiness Score",              
        hover_data={"Happiness Score": True},
        color_continuous_scale='thermal'
    )

    fig.update_layout(
        template="plotly_white",
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='natural earth', fitbounds="locations"
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        width=600,
        coloraxis_colorbar=dict(
            x=1,  
            y=0.5,
            len=1,
            thickness=20,
            outlinewidth=0,
            title=""
        )
    )

    return fig


def media_graph(data, years, year, country=None):
    df = data[data['year'] <= year].copy()
    a, b = years
    df = df[(df['year'] >= a) & (df['year'] <= b)]

    df_mean_global = df.groupby('year', as_index=False)['Happiness Score'].mean()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_mean_global['year'],
        y=df_mean_global['Happiness Score'],
        mode='lines+markers',
        line=dict(color='#7e4d8f', width=3, dash='dash' if country is not None else 'solid'),
        marker=dict(color='#7e4d8f', size=8),
        hovertemplate='Media global<br>Año: %{x}<br>Valor: %{y:.2f}<extra></extra>',
        showlegend=False,

    ))

    if country is not None:
        df_country = df[df['Country'] == country]
        df_mean_country = df_country.groupby('year', as_index=False)['Happiness Score'].mean()
        fig.add_trace(go.Scatter(
            x=df_mean_country['year'],
            y=df_mean_country['Happiness Score'],
            mode='lines+markers',
            name=country,
            line=dict(color='#f79343', width=3),
            marker=dict(color='#f79343', size=8),
            hovertemplate='Media ' + country + '<br>Año: %{x}<br>Valor: %{y:.2f}<extra></extra>',
            showlegend=False
        ))

    fig.update_xaxes(showticklabels=False, title_text='', dtick=1)
    fig.update_yaxes(showticklabels=False, title_text='')

    fig.update_layout(
        template='plotly_white',
        margin=dict(l=0, r=0, t=0, b=0),
        height=150,
        width=300,
    )

    y_max = df_mean_global['Happiness Score'].max()
    y_min = df_mean_global['Happiness Score'].min()
    if country is not None:
        y_max = max(y_max, df_mean_country['Happiness Score'].max())
        y_min = min(y_min, df_mean_country['Happiness Score'].min())

    text_y = y_min - 0.3

    fig.add_trace(go.Scatter(
        x=df_mean_global['year'],
        y=[text_y] * len(df_mean_global),
        mode='text',
        text=[str(y) for y in df_mean_global['year']],
        textposition='bottom center',
        textfont=dict(color='black', size=8),
        showlegend=False,
        hoverinfo='skip',
        hovertemplate=''
    ))

    fig.update_yaxes(range=[y_min - 0.5, y_max * 1.05])

    fig.update_layout(yaxis=dict(scaleanchor=None, automargin=True))
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=20))

    return fig


def paises_graph(data, years, year, country=None):
    df = data[data['year'] <= year].copy()
    a, b = years
    df = df[(df['year'] >= a) & (df['year'] <= b)]

    df_mean = df.groupby('year', as_index=False)['Happiness Score'].mean()
    df_mean.rename(columns={'Happiness Score': 'mean_score'}, inplace=True)
    df_with_mean = df.merge(df_mean, on='year')

    if country is not None:
        df_with_mean = df_with_mean[df_with_mean['Country'] == country]
        df_with_mean['count_up'] = (df_with_mean['Happiness Score'] >= df_with_mean['mean_score']).astype(int)
        df_with_mean['count_down'] = (df_with_mean['Happiness Score'] < df_with_mean['mean_score']).astype(int)
        df_counts_up = df_with_mean[['year', 'count_up']].rename(columns={'count_up': 'count'})
        df_counts_down = df_with_mean[['year', 'count_down']].rename(columns={'count_down': 'count'})
    else:
        df_counts_up = df_with_mean.groupby('year')['Happiness Score'].apply(
            lambda x: (x >= df_with_mean.loc[x.index, 'mean_score']).sum()
        ).reset_index(name='count')
        df_counts_down = df_with_mean.groupby('year')['Happiness Score'].apply(
            lambda x: (x < df_with_mean.loc[x.index, 'mean_score']).sum()
        ).reset_index(name='count')

    if df_counts_up.empty and df_counts_down.empty:
        print("No hay datos para los años seleccionados.")
        return

    def get_colors(values, inverted=False):
        scale = px.colors.sequential.thermal
        if inverted:
            scale = scale[::-1]
        return [scale[int(v / 100 * (len(scale) - 1))] for v in values]

    colors_up = ['#f79343' if c == 1 else get_colors(df_counts_up['count']) for c in df_counts_up['count']]
    colors_down = ['#7e4d8f' if c == 1 else get_colors(df_counts_down['count'], inverted=True) for c in df_counts_down['count']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_counts_up['year'],
        y=df_counts_up['count'],
        text=[f"<b>{c}</b>" for c in df_counts_up['count']],
        textposition='inside',
        textfont=dict(color='white', size=50),
        marker_color=colors_up,
        name='>= media',
        offsetgroup=0, width=0.8,
        hoverinfo='skip',
        hovertemplate=''
    ))

    fig.add_trace(go.Bar(
        x=df_counts_down['year'],
        y=[-c for c in df_counts_down['count']],
        text=[f"<b>{c}</b>" for c in df_counts_down['count']],
        textposition='inside',
        textfont=dict(color='white', size=50),
        marker_color=colors_down,
        name='< media',
        offsetgroup=0, width=0.8,
        hoverinfo='skip',
        hovertemplate=''
    ))

    y_max = max(df_counts_up['count'].max(), 0)
    y_min = -df_counts_down['count'].max() if not df_counts_down.empty else 0

    fig.update_yaxes(range=[y_min - 1, y_max + 1])
    fig.update_xaxes(showticklabels=False, title_text='', dtick=1)
    fig.update_yaxes(showticklabels=False, title_text='')

    fig.update_layout(
        template='plotly_white',
        margin=dict(l=0, r=0, t=20, b=20),
        showlegend=False,
        height=150,
        width=300,
    )

    text_y = -0.3 if country is None else y_min - 0.3
    color_text = "white" if country is None else "black"

    fig.add_trace(go.Scatter(
        x=df_counts_up['year'],
        y=[text_y] * len(df_counts_up),
        mode='text',
        text=[str(y) for y in df_counts_up['year']],
        textposition='bottom center',
        textfont=dict(color=color_text, size=8),
        showlegend=False,
        hoverinfo='skip',
        hovertemplate=''
    ))


    return fig


def media_bar_region(data, year, use_aggregate_region=True):

    region_col = 'Aggregate Region' if use_aggregate_region else 'Region'
    agrupacion = 'continente' if use_aggregate_region else 'región'
    df_year = data[data['year'] == year].copy()

    if df_year.empty:
        print(f"No hay datos para el año {year}")
        return

    mean_global = df_year['Happiness Score'].mean()

    df_region_mean = df_year.groupby(region_col, as_index=False)['Happiness Score'].mean()
    df_region_mean = df_region_mean.sort_values('Happiness Score', ascending=False)

    colors_palette = px.colors.sequential.thermal 
    n_colors = len(colors_palette)

    a,b=[df_region_mean['Happiness Score'].min()-1,df_region_mean['Happiness Score'].max()+1]

    bar_colors = [
        colors_palette[int(max(0, min(n_colors-1, (score - a) / (b - a) * (n_colors - 1))))]
        for score in df_region_mean['Happiness Score']
    ]

    x_labels = [name.replace(" ", "<br>") for name in df_region_mean[region_col]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x_labels,
        y=df_region_mean['Happiness Score'],
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in df_region_mean['Happiness Score']],
        textposition='inside',
        textfont=dict(color='white', size=12),
        name='Regiones', width=0.6, showlegend=False,
        customdata=df_region_mean[[region_col, 'Happiness Score']],
        hovertemplate='Region: %{customdata[0]}<br>Happiness: %{customdata[1]:.2f}<extra></extra>',

    ))

    fig.add_trace(go.Scatter(
        x=x_labels,
        y=[mean_global]*len(df_region_mean),
        mode='lines+text',
        line=dict(color='#7e4d8f', width=3, dash='dash'),
        text=[None]*(len(df_region_mean)-1) + [f"{mean_global:.2f}"], 
        textposition='middle right',
        textfont=dict(color='#7e4d8f', size=12),
        name='Media Global', showlegend=False,
        hoverinfo='skip', 
        hovertemplate=''
    ))

    fig.update_layout(
        template='plotly_white',
        title=dict(
            text=f"Medias de Índice de Felicidad por {agrupacion} en {year}",
            x=0.5, xanchor='center',
            font=dict(family="Segoe UI", size=17, color="black")
        ),
        xaxis=dict(title=''),  
        yaxis=dict(title='', showticklabels=False), 
        margin=dict(l=10, r=10, t=60, b=0),
        height=200,
        width=600,
        showlegend=False
    )

    return fig

def media_line_region(data, years, year, region="Europe", use_aggregate_region=True):
    df = data[data['year'] <= year].copy()
    a, b = years
    df = df[(df['year'] >= a) & (df['year'] <= b)]

    df_mean_global = df.groupby('year', as_index=False)['Happiness Score'].mean()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_mean_global['year'],
        y=df_mean_global['Happiness Score'],
        mode='lines+markers+text',
        line=dict(color='#7e4d8f', width=3, dash='dash'),
        marker=dict(color='#7e4d8f', size=8),
        text=[f"{v:.2f}" for v in df_mean_global['Happiness Score']],
        textposition='top center',
        textfont=dict(color='#7e4d8f', size=12),
        hoverinfo='skip',
        hovertemplate='', showlegend=False,
    ))

    columna = "Aggregate Region" if use_aggregate_region else "Region"
    df_region = df[df[columna] == region]
    df_mean_region = df_region.groupby('year', as_index=False)['Happiness Score'].mean()

    fig.add_trace(go.Scatter(
        x=df_mean_region['year'],
        y=df_mean_region['Happiness Score'],
        mode='lines+markers+text',
        name=region,
        line=dict(color='#f79343', width=3),
        marker=dict(color='#f79343', size=8),
        text=[f"{v:.2f}" for v in df_mean_region['Happiness Score']],
        textposition='top center',
        textfont=dict(color='#f79343', size=12),
        hoverinfo='skip',
        hovertemplate='', showlegend=False,
    ))

    fig.update_xaxes(showticklabels=False, title_text='', dtick=1)
    fig.update_yaxes(showticklabels=False, title_text='')

    fig.update_layout(
        template='plotly_white',
        margin=dict(l=0, r=0, t=20, b=20),
        height=150,
        width=600,
        title=dict(
            text=f"Evolución de la media del Índice de Felicidad{' de ' + region if region is not None else ''}",
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(family="Segoe UI", size=17, color="black")
        )
    )

    y_max = max(df_mean_global['Happiness Score'].max(), df_mean_region['Happiness Score'].max())
    y_min = min(df_mean_global['Happiness Score'].min(), df_mean_region['Happiness Score'].min())

    text_y = y_min - 0.5

    fig.add_trace(go.Scatter(
        x=df_mean_global['year'],
        y=[text_y] * len(df_mean_global),
        mode='text',
        text=[str(y) for y in df_mean_global['year']],
        textposition='bottom center',
        textfont=dict(color='black', size=8),
        showlegend=False,
        hoverinfo='skip',
        hovertemplate=''
    ))

    fig.update_yaxes(range=[y_min - 0.8, y_max * 1.1])

    return fig


def radar_factores_region(data, years, agrupar=True, normalizar_por_region=False):
    def hex_to_rgba(hex_color, alpha=0.2):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'

    a, b = years
    df = data[(data['year'] >= a) & (data['year'] <= b)].copy()

    factores = [
        "GDP per Capita",
        "Social Support",
        "Healthy Life Expectancy",
        "Freedom",
        "Perceptions of Corruption",
        "Generosity"
    ]

    col_group = "Aggregate Region" if agrupar else "Region"
    df_mean = df.groupby(col_group)[factores].mean().reset_index()

    if not normalizar_por_region:
        max_val = df[factores].max().max()
        df_mean[factores] = df_mean[factores] / max_val * 100

    thermal_palette = colors.sequential.thermal

    def rgb_to_hex(c):
        if c.startswith('rgb'):
            parts = list(map(int, c[4:-1].split(',')))
            return '#{:02x}{:02x}{:02x}'.format(*parts)
        return c

    def sampled_colors(n, palette):
        total = len(palette)
        indices = np.arange(total)
        intercalados = np.concatenate([indices[::2], indices[1::2]])
        chosen_indices = [intercalados[i % total] for i in range(n)]
        return [rgb_to_hex(palette[i]) for i in chosen_indices]

    num_regiones = df_mean.shape[0]

    colores = sampled_colors(num_regiones, thermal_palette)

    df_mean['area'] = df_mean[factores].sum(axis=1)
    df_mean = df_mean.sort_values('area', ascending=False)

    fig = go.Figure()

    for i, row in df_mean.iterrows():
        valores = row[factores].tolist()
        if normalizar_por_region:
            max_region = max(valores)
            if max_region > 0:
                valores = [v / max_region * 100 for v in valores]
        valores.append(valores[0])
        color = colores[i % len(colores)]
        fig.add_trace(go.Scatterpolar(
            r=valores,
            theta=factores + [factores[0]],
            fill='toself',
            fillcolor=hex_to_rgba(color, 0),
            line=dict(color=color, width=3),
            marker=dict(size=6, color=color),
            name=row[col_group],
            hovertemplate="<b>%{theta}</b><br>Valor medio: %{r:.2f}<extra>%{text}</extra>",
            text=row[col_group],
        ))

    valor_maximo = 100 if normalizar_por_region else df_mean[factores].max().max() * 1.05

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, valor_maximo],
                showticklabels=False,
                gridcolor='lightgrey'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, family='Segoe UI', color='black')
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1 if agrupar else 1.3,
            y=1.0,
            xanchor='left',
            yanchor='top',
            font=dict(family='Segoe UI', size=12)
        ),
        template='plotly_white',
        margin=dict(l=0, r=0, t=80, b=0),
        title=dict(
            text=f"Media de Factores del Índice de Felicidad por {'continente' if agrupar else 'región'} ({a}–{b})",
            x=0.5,
            xanchor='center',
            font=dict(family="Segoe UI", size=17, color="black"),
        ),
        width=600,
        height=500,
    )

    return fig

def grafica_rank(data, top_n, years):
    from plotly import colors
    import plotly.graph_objects as go

    data_filtered = data[(data["year"] >= years[0]) & (data["year"] <= years[1])]

    top_countries_year = (
        data_filtered.groupby("year")
        .apply(lambda df: df.nsmallest(top_n, "Happiness Rank"))
        .reset_index(drop=True)
    )

    countries_visible = sorted(top_countries_year["Country"].unique())
    palette = colors.sequential.thermal
    color_map = {country: palette[i % len(palette)] for i, country in enumerate(countries_visible)}

    fig = go.Figure()

    def add_line_segment(fig, df, color, width=2, dash=None, opacity=1):
        fig.add_trace(go.Scatter(
            x=df["year"],
            y=df["Happiness Rank"],
            mode="lines+markers",
            line=dict(color=color, width=width, dash=dash),
            marker=dict(opacity=0),
            opacity=opacity,
            showlegend=False,
            hovertemplate="<b>País:</b> %{customdata[0]}<br><b>Año:</b> %{x}<br><b>Ranking:</b> %{y}<extra></extra>",
            customdata=[[c] for c in df["Country"]]  
        ))

    for country in countries_visible:
        df = top_countries_year[top_countries_year["Country"] == country].sort_values("year")
        for i in range(len(df) - 1):
            gap = df["year"].iloc[i + 1] - df["year"].iloc[i]
            dash_style = None if gap == 1 else "dash"
            add_line_segment(fig, df.iloc[i:i + 2], color=color_map[country], dash=dash_style)

    for country in countries_visible:
        first = top_countries_year[top_countries_year["Country"] == country].sort_values("year").iloc[0]
        fig.add_annotation(
            x=first["year"], y=first["Happiness Rank"], text=country,
            showarrow=False, xshift=-40,
            font=dict(size=12, color=color_map[country], family="Segoe UI"),
            xanchor="center"
        )

    fig.update_yaxes(autorange="reversed", showticklabels=False, title_text=None)
    fig.update_layout(
        template="plotly_white",
        title=f"Top {top_n} países en Índice de Felicidad ({years[0]}–{years[1]})",
        title_font=dict(family="Segoe UI", size=17, color="black"),
        font=dict(family="Segoe UI", size=12, color="black"),
        title_x=0.5,
        xaxis_title=None,
        xaxis=dict(tickmode='linear', dtick=1, tickfont=dict(family="Segoe UI", size=8)),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
        height=500,
        width=600,
    )

    for country in countries_visible:
        df = top_countries_year[top_countries_year["Country"] == country].sort_values("year")
        years_c = df["year"].values
        for i, y in enumerate(years_c):
            prev = years_c[max(i-2, 0):i]
            nxt = years_c[i+1:i+3]
            if all(abs(y - p) > 1 for p in prev) and all(abs(n - y) > 1 for n in nxt):
                fig.add_trace(go.Scatter(
                    x=[y],
                    y=[df["Happiness Rank"].iloc[i]],
                    mode="markers",
                    marker=dict(size=5, color=color_map[country]),
                    showlegend=False,
                ))

    return fig


def happiness_forecast(data, years, year, country=None, number_years=3):
    df = data[data['year'] <= year].copy()
    df = df[df['year'].between(years[0], years[1])]

    if country is None:
        df = df.groupby('year', as_index=False)['Happiness Score'].mean()
    else:
        df = df[df['Country'] == country]

    x_hist = df['year'].values.reshape(-1,1)
    y_hist = df['Happiness Score'].values

    model = LinearRegression()
    model.fit(x_hist, y_hist)

    future_years = np.arange(year+1, year+1+number_years)
    y_pred = model.predict(future_years.reshape(-1,1))

    std_dev = y_hist.std() 
    horizon_factor = np.linspace(1, 1 + number_years*0.2, number_years) 
    deltas = std_dev * horizon_factor

    y_upper = y_pred + deltas
    y_lower = y_pred - deltas

    x_connect = [year, future_years[0]]
    y_connect = [y_hist[-1], y_pred[0]]

    y_min = min(y_hist.min(), y_lower.min()) - 0.5
    y_max = max(y_hist.max(), y_upper.max()) + 0.5

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_hist.flatten(),
        y=y_hist,
        mode='lines+markers+text',
        line=dict(color='#7e4d8f', width=3),
        marker=dict(color='#7e4d8f', size=6),
        text=[f"{v:.2f}" for v in y_hist],
        textposition='top center',
        textfont=dict(family='Segoe UI', size=14, color='#7e4d8f'),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=x_connect,
        y=y_connect,
        mode='lines',
        line=dict(color='#f79343', width=2),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=future_years,
        y=y_pred,
        mode='lines+markers+text',
        line=dict(color='#f79343', width=3, dash='dash'),
        marker=dict(color='#f79343', size=6),
        text=[f"{v:.2f}" for v in y_pred],
        textposition='top center',
        textfont=dict(family='Segoe UI', size=14, color='#f79343'),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([future_years, future_years[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(247, 147, 67, 0.2)',
        line=dict(color='rgba(247, 147, 67, 0.3)'),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=future_years,
        y=y_upper,
        mode='lines',
        line=dict(color='#f79343', width=2, dash='dot'),
        hoverinfo='skip',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=future_years,
        y=y_lower,
        mode='lines',
        line=dict(color='#f79343', width=2, dash='dot'),
        hoverinfo='skip',
        showlegend=False
    ))

    all_years = list(x_hist.flatten()) + list(future_years)
    fig.add_trace(go.Scatter(
        x=all_years,
        y=[y_min+0.2]*len(all_years),
        mode='text',
        text=[str(y) for y in all_years],
        textposition='bottom center',
        textfont=dict(family='Segoe UI', size=8, color='black'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(range=[y_min, y_max], showticklabels=False, title_text='')

    title_text = (
        f"Evolución y predicción del Índice de Felicidad de {country}"
        if country is not None
        else "Evolución y predicción del Índice de Felicidad global"
    )


    fig.update_layout(
        template='plotly_white',
        title=dict(
            text=title_text,
            font=dict(family='Segoe UI', size=17, color='black'),
            x=0.5, xanchor='center'
        ),
        width=600,
        height=200,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    fig.add_trace(go.Scatter(
        x=[year, future_years[0]],
        y=[y_hist[-1], y_upper[0]],
        mode='lines',
        line=dict(color='#f79343', width=2, dash='dot'),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[year, future_years[0]],
        y=[y_hist[-1], y_lower[0]],
        mode='lines',
        line=dict(color='#f79343', width=2, dash='dot'),
        hoverinfo='skip',
        showlegend=False
    ))

    x_area = [year, future_years[0], future_years[0], year]
    y_area = [y_hist[-1], y_upper[0], y_lower[0], y_hist[-1]]

    fig.add_trace(go.Scatter(
        x=x_area,
        y=y_area,
        fill='toself',
        fillcolor="rgba(247, 147, 67, 0.2)",
        line=dict(color='rgba(247, 147, 67, 0.2)'),
        hoverinfo='skip',
        showlegend=False
    ))

    return fig
