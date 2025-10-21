import pandas as pd           
import numpy as np          
from collections import Counter  

import plotly.graph_objects as go  
import plotly.express as px        


def grafica_rank(data, top_n, years):
    color_1="#E6C32C"
    color_2="#2C2DE6"

    data_filtered = data[(data["year"] >= years[0]) & (data["year"] <= years[1])]

    top_countries_year = (
        data_filtered.groupby("year")
        .apply(lambda df: df.nsmallest(top_n, "Happiness Rank"))
        .reset_index(drop=True)
    )

    countries_visible = sorted(top_countries_year["Country"].unique())
    palette = px.colors.qualitative.Set2
    color_map = {country: palette[i % len(palette)] for i, country in enumerate(countries_visible)}

    region_colors = ["#E74C3C", "#27AE60", "#FFFBAF", "#8E44AD", "#74CEE9"]
    regions_visible = sorted(top_countries_year["Aggregate Region"].unique())
    region_color_map = {r: region_colors[i] for i, r in enumerate(regions_visible)}

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
            hovertemplate="<b>Country:</b> %{customdata[0]}<br><b>Year:</b> %{x}<br><b>Rank:</b> %{y}<extra></extra>",
            customdata=[[c] for c in df["Country"]]  

        ))

    region_counts = Counter(top_countries_year["Region"])
    major_region, count = region_counts.most_common(1)[0]
    percent = round(count / len(top_countries_year) * 100, 1)

    fig.add_annotation(
        text=f'<span style="color:{color_1};">{percent}%</span> of all countries across all years in the top {top_n} are from <span style="color:{color_2};">{major_region}</span>',
        xref="paper", yref="paper",
        x=0.53, y=1.13, showarrow=False,
        font=dict(color="black", size=15)
    )

    for country in countries_visible:
        df = top_countries_year[top_countries_year["Country"] == country].sort_values("year")
        region = df["Aggregate Region"].iloc[0]
        if region != "Europe":
            for i in range(len(df) - 1):
                if df["year"].iloc[i + 1] - df["year"].iloc[i] == 1:
                    segment = df.iloc[i:i + 2]
                    fig.add_trace(go.Scatter(
                        x=segment["year"],
                        y=segment["Happiness Rank"],
                        mode="lines",
                        line=dict(color=region_color_map[region], width=15),
                        opacity=0.25,
                        showlegend=False,
                        hoverinfo="skip"
                    ))

    for country in countries_visible:
        df = top_countries_year[top_countries_year["Country"] == country].sort_values("year")
        for i in range(len(df) - 1):
            gap = df["year"].iloc[i + 1] - df["year"].iloc[i]
            if gap in [1, 2]:
                dash_style = None if gap == 1 else "dash"
                add_line_segment(fig, df.iloc[i:i + 2], color=color_map[country], dash=dash_style)

    for country in countries_visible:
        first = top_countries_year[top_countries_year["Country"] == country].sort_values("year").iloc[0]
        fig.add_annotation(
            x=first["year"], y=first["Happiness Rank"], text=country,
            showarrow=False, xshift=-40,
            font=dict(size=12, color=color_map[country]),
            xanchor="center"
        )

    fig.update_yaxes(autorange="reversed", showticklabels=False, title_text=None)
    fig.update_layout(
        template="plotly_white",
        title=f"Top {top_n} Countries Happiness Rank ({years[0]}–{years[1]})",
        title_x=0.5,
        font_size=12,
        xaxis_title=None,
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=True,
        legend_title_text="Region"
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

    for region in [r for r in regions_visible if r != "Europe"]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=region_color_map[region], width=15),
            opacity=0.4,
            name=region, showlegend=True
        ))

    if "Europe" in regions_visible:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="#FFFFFF", width=15),
            name="Europe", showlegend=True
        ))


    fig.update_xaxes(tickmode='linear', dtick=1)

    return fig


def score_by_region(data, violin=False, show_means=True, group_by_region_agregada=True, years=[2015,2022]):

    original_years=years

    data = data[(data['year'] >= years[0]) & (data['year'] <= years[1])]


    group_col = "Aggregate Region" if group_by_region_agregada else "Region"
    regions = data[group_col].unique()
    palette = px.colors.qualitative.Set2
    region_color_map = {r: palette[i % len(palette)] for i, r in enumerate(regions)}
    regions = data.groupby(group_col)["Happiness Score"].mean().sort_values(ascending=False).index.tolist()

    fig = go.Figure()
    max_years = data.groupby(group_col)["year"].nunique().max()
    year_offsets = np.linspace(-0.3, 0.3, max_years)
    years = sorted(data["year"].unique())

    for region in regions:
        df_r = data[data[group_col] == region]
        df_years = sorted(df_r["year"].unique())
        x_line, y_line = [], []

        for i, y in enumerate(df_years):
            df_y = df_r[df_r["year"] == y]
            if df_y.empty:
                continue
            mean, mn, mx = df_y["Happiness Score"].mean(), df_y["Happiness Score"].min(), df_y["Happiness Score"].max()
            x_pos = regions.index(region) + year_offsets[i] * 1.5
            c = region_color_map[region]

            if not violin:
                fig.add_trace(go.Scatter(x=[x_pos, x_pos], y=[mn, mx], mode="lines", line=dict(color=c, width=8), opacity=0.25, showlegend=False, hoverinfo="skip"))
            else:
                fig.add_trace(go.Violin(x=[x_pos]*len(df_y), y=df_y["Happiness Score"], line_color=c, fillcolor=c, opacity=0.25, width=0.05, points=False, showlegend=False, hoverinfo="skip"))

            fig.add_trace(go.Scatter(x=[x_pos], y=[mean], mode="markers", marker=dict(size=12, color=c), showlegend=False,
                                    hovertemplate=f"{group_col}: {region}<br>Year: {y}<br>Mean: {mean:.2f}<br>Min: {mn:.2f}<br>Max: {mx:.2f}<extra></extra>"))
            x_line.append(x_pos)
            y_line.append(mean)

        if len(x_line) > 1:
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color=c, width=2), name=region, showlegend=False, hoverinfo="skip"))

        if show_means:
            mean_all = df_r["Happiness Score"].mean()
            fig.add_trace(go.Scatter(x=[-0.5, len(regions)-0.5], y=[mean_all, mean_all], mode="lines", line=dict(color=c, width=6), opacity=0.25, showlegend=False, hoverinfo="skip"))


    fig.update_layout(
        template="plotly_white",
        title=f"Happiness Score by {'Aggregate Region' if group_by_region_agregada else 'Region'} ({original_years[0]}-{original_years[1]})",
        title_x=0.5,
        font_size=12,
        xaxis_title=None,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(regions))),
            ticktext=[r.replace(' ', '<br>') for r in regions]  
        ),
        margin=dict(l=50, r=50, t=100, b=50),
        legend_title_text=f"{'Aggregate Region' if group_by_region_agregada else 'Region'}",
    )



    return fig

factors = ['GDP per Capita','Social Support','Healthy Life Expectancy','Freedom',
           'Perceptions of Corruption','Generosity']

def peso_por_factor(data, group_by_region_agregada, years):

    color_2="#2C2DE6"

    group_col = 'Aggregate Region' if group_by_region_agregada else 'Region'
    df = data[(data['year'] >= years[0]) & (data['year'] <= years[1])]
    df_grouped = df.groupby(group_col)[factors].mean().reset_index()

    min_val, max_val = df_grouped[factors].min().min(), df_grouped[factors].max().max()
    min_size, max_size = 10, 60
    for f in factors:
        df_grouped[f'{f}_Size'] = ((df_grouped[f] - min_val) / (max_val - min_val)) * (max_size - min_size) + min_size

    region_order = df_grouped.sort_values('GDP per Capita', ascending=False)[group_col].tolist()
    df_grouped[group_col] = pd.Categorical(df_grouped[group_col], categories=region_order, ordered=True)

    palette = px.colors.qualitative.Set2
    color_map = {r: palette[i % len(palette)] for i, r in enumerate(df_grouped[group_col])}
    df_grouped['Color'] = df_grouped[group_col].map(color_map)


    def get_text_color(rgb_string, size):
        if size < 30:
            return 'black'
        rgb = [int(x) for x in rgb_string.strip('rgb()').split(',')]
        lum = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        return 'white' if lum < 128 else 'black'

    fig = go.Figure()

    factors_sorted = df_grouped[factors].mean().sort_values(ascending=False).index.tolist()
    for _, row in df_grouped.iterrows():
        for f in factors_sorted:
            sz = row[f'{f}_Size']
            fig.add_trace(go.Scatter(
                x=[f], y=[row[group_col]], mode='markers+text',
                marker=dict(size=sz, color=row['Color'], line=dict(color='white', width=1)),
                text=[f"{row[f]:.2f}"],
                textposition='top center' if sz < 30 else 'middle center',
                textfont=dict(color=get_text_color(row['Color'], sz), size=10),
                hovertemplate=f"<b>{group_col}:</b> {row[group_col]}<br><b>Factor:</b> {f}<br><b>Media:</b> {row[f]:.2f}<extra></extra>",
                showlegend=False
            ))

    height = 600
    if not group_by_region_agregada:
        height += 20 * len(df_grouped)

    fig.update_layout(
        template="plotly_white",
        title=f"Average Factor Contribution by Region ({years[0]}–{years[1]})",
        title_x=0.5,
        font_size=12,
        xaxis_title=None,
        margin=dict(l=50, r=50, t=100, b=50),
        )

    annotations_text = "".join([
        f'<span style="color:black;">{factor}</span> = <span style="color:{color_2};">{df_grouped[factor].mean():.2f}</span>'
        + ("<br>" if (i + 1) % 3 == 0 else ", ")
        for i, factor in enumerate(factors_sorted)
    ])


    fig.add_annotation(
        text=annotations_text,
        xref="paper", yref="paper",
        x=(0.5 if group_by_region_agregada else 0.42), y= (1.08 if group_by_region_agregada else 1.04),
        showarrow=False,
        font=dict(size=12),
        align="left"
    )

    fig.update_layout(
        yaxis=dict(
            automargin=True,
            tickmode='array',
            tickvals=list(range(len(df_grouped))),
            ticktext=df_grouped[group_col],
            tickfont=dict(size=12),
            categoryorder='array',
            categoryarray=df_grouped[group_col]
        ),
        height=height + (10 if not group_by_region_agregada else 1) * len(df_grouped)
    )


    return fig

def scatter_happiness(data, group_by_region_agregada=True, years=[2015, 2024], factor='GDP per Capita', show_trend=True, show_points=True):

    color_2="#2C2DE6"

    group_col = "Aggregate Region" if group_by_region_agregada else "Region"
    df = data[(data['year'] >= years[0]) & (data['year'] <= years[1])]

    regions = df[group_col].unique()
    palette = px.colors.qualitative.Set2
    color_map = {r: palette[i % len(palette)] for i, r in enumerate(regions)}

    fig = go.Figure()
    for region in regions:
        df_r = df[df[group_col] == region]
        if show_points:
            fig.add_trace(go.Scatter(
                x=df_r[factor],
                y=df_r['Happiness Score'],
                mode='markers',
                marker=dict(size=5, color=color_map[region], line=dict(color='white', width=1)),
                name=region,
                showlegend=True,
                hoverinfo='skip'
            ))
        df_r = df_r.dropna(subset=[factor, 'Happiness Score'])
        if show_trend and len(df_r) > 1 and df_r[factor].nunique() > 1:
            coeffs = np.polyfit(df_r[factor], df_r['Happiness Score'], 1)
            trend = np.poly1d(coeffs)
            x_vals = np.linspace(df_r[factor].min(), df_r[factor].max(), 100)
            fig.add_trace(go.Scatter(x=x_vals, y=trend(x_vals), mode='lines',
                        line=dict(color=color_map[region], width=10), opacity=0.3,
                        name=f"{region} trend", hoverinfo='skip'))

            r = df_r[factor].corr(df_r['Happiness Score'])

            fig.add_trace(go.Scatter(
                x=[x_vals[-1]],
                y=[trend(x_vals[-1])],
                mode='text',
                text=[f"  {r:.2f}"],  
                textposition="middle right",
                textfont=dict(color=color_map[region], size=15),
                showlegend=False,
                hoverinfo='skip'
            ))

    overall_r = df[factor].corr(df['Happiness Score'])
    annotations_text = f'<span style="color:"black";">Overall correlation = </span> <span style="color:{color_2};">{round(overall_r, 2)}</span>'

    


    fig.update_layout(
        template="plotly_white",
        title=f"Happiness Score vs {factor} ({years[0]}–{years[1]})",
        title_x=0.5,
        font_size=12,
        xaxis_title=factor,
        yaxis_title="Happiness Score",
        margin=dict(l=50, r=50, t=100, b=50),
    )

    fig.add_annotation(
        text=annotations_text,
        xref="paper", yref="paper",
        x=(0.54 if group_by_region_agregada else 0.6), y=(1.12 if group_by_region_agregada else 1.12),
        showarrow=False,
        font=dict(size=12),
        align="left"
    )

    return fig

def happiness_map(data, year, projection_type, color_scale='RdYlGn'):
    df_year = data[data['year'] == year].copy()

    fig = px.choropleth(
        df_year,
        locations="Country",
        locationmode="country names",
        color="Happiness Score",              
        hover_name="Country",
        hover_data={"Happiness Score": True},
        color_continuous_scale=color_scale,   
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Happiness Score by Country ({year})",
        title_x=0.5,
        font_size=12,
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type=projection_type
        )
    )

    return fig

