import dash
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import json
import plotly
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots
import chardet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import euclidean_distances
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
app = dash.Dash()
import numpy as np
with open("./original_police_Killings.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
df = pd.read_csv("./original_police_Killings.csv", encoding="ISO-8859-1")
df.head()
app.layout = html.Div(className='layout', children=[
    html.Nav(className="Navbar", children=[

    dcc.Dropdown(id="options", options=[
            {'label': 'Correlation Matrix', 'value': 'correlation'},
            {'label': '5×5 scatter plot matrix', 'value': 'scatterplot'},
            {'label': 'parallel coordinates', 'value': 'parallel'},
            {'label': 'PCA plot', 'value': 'pca'},
            {'label': 'biplot', 'value': 'biplot'},
            {'label': 'MDS display(euclidian)', 'value': 'msd1'},
            {'label': 'MDS display(1-|correlation|)', 'value': 'msd2'},
           
        ],
            value='correlation',
            multi=False,
            clearable=False,
            style={"width": "50%"})

    ]),
    html.Div(className="graph_section", children=[
        dcc.Graph(id="graph")
    ]), 
    html.Div(id="scree_plot", children=[
        dcc.Graph(id="scree_graph")
    ])
])

@app.callback(
    [Output(component_id='graph', component_property='figure'), Output('scree_plot', 'style'),Output(component_id='scree_graph', component_property='figure')],
    [Input(component_id='options', component_property='value')]
)
def display_graph(options):
    if options == "correlation":
        return display_correlation(),  {'display':'none'},  dash.no_update 
    elif options == "scatterplot":
        return display_scatterplot(),  {'display':'none'},  dash.no_update 
    elif options == "parallel":
        return display_parallel(),  {'display':'none'},  dash.no_update 
    elif options == 'pca':
        return display_pca(),  {'display':'block'},  get_scree()
    elif options == 'biplot':
        return display_biplot(),  {'display':'none'},  dash.no_update
    elif options == 'msd1':
        return display_mds_euclidian(),  {'display':'none'},  dash.no_update
    elif options == 'msd2':
        return display_mds_second(),  {'display':'none'},  dash.no_update
def display_mds_second():
    figure = None
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    correlated_data = df2.corr(method='pearson').abs()
    similarities = 1 - correlated_data
    embedding = MDS(n_components=2, metric=True,  dissimilarity = "precomputed")
    df_transformed = embedding.fit_transform(similarities)
    print(df_transformed)
    # mds_data = MDS(n_components=2, dissimilarity='precomputed')
    # similarity = pairwise_distances(df2, metric='correlation')
    # originalMDSCo = mds_data.fit_transform(similarity)
    figure= px.scatter(df_transformed,x = 0, y = 1 , height= 500, color=1 ,color_continuous_scale= "RdBu", color_continuous_midpoint=1)
    figure.update_layout(legend_title="Legend Title", title ="MDS PLOT 1 -|correlation|")
    return figure
def display_mds_euclidian():
    figure = None
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    embedding = MDS(n_components=2, metric=True,  dissimilarity = "euclidean")
    df_transformed = embedding.fit_transform(df2)
    # print(df_transformed)
    figure= px.scatter(df_transformed, height=500, color= 1, x =0, y =1)
    figure.update_layout(legend_title="Legend Title", title ="MDS PLOT EUCLEDIAN")
    return figure
def display_biplot():
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    x = StandardScaler().fit_transform(df2)
    pca = PCA(n_components=10)
    components = pca.fit_transform(x)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    figure= px.scatter(components, x=0, y=1, height=500 ,  color=df['raceethnicity'], title="Biplot")
    for i in range(10):
        figure.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1]
        )
    return figure
def get_scree():
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    x = StandardScaler().fit_transform(df2)
    pca = PCA(n_components=10)
    pca.fit_transform(x)
    indices = ['1', '2', '3', '4', '5', '6', '7',  '8',  '9',  '10']
    eigenVals = []
    for i in pca.explained_variance_:
        eigenVals.append(i)
    y = pd.DataFrame(list(zip( eigenVals, indices,)), columns=[ "Eigen Values","Index"])
    return px.bar(y, x='Index', y = 'Eigen Values', title ="Scree Plot")
def display_pca():
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    
    x = StandardScaler().fit_transform(df2)
    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    figure= px.scatter(components,height=500,  x=0, y=1, color=df['raceethnicity'], title="PCA Plot")
    for i in range(2):
        figure.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1]
        )
        
    return figure
def display_parallel():
    figure =None
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    correlated_data = df2.corr(method='pearson').abs().sum(axis=0).sort_values(ascending=False, inplace=False)
    A1 = correlated_data[0]
    correlated_data= df2.corr(method='pearson')
    df3 = df2[["pov", "h_income", "college", "urate",'share_black', 'share_white',  'share_hispanic', 'age', 'pop', 'county_income']]
    figure = px.parallel_coordinates(df3,title ="Parallel Coordinates",  height=500 ,color_continuous_scale =None,range_color =[1,2], color_continuous_midpoint=2, 
    labels={"pov":"Poverty Rate", "h_income":"House Hold Income", "college":"College Rate", "urate":"Unemployment Rate", "share_black":"Black Population", "share_white":"White Population", "share_hispanic":"Hispanic Population", "age":"Age", "pop":"Population", "county_income":"County Income"})
    return figure

def display_scatterplot():
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    correlated_data = df2.corr(method='pearson')
    absolute_values = correlated_data.abs()
    sum_values = absolute_values.sum(axis=0).sort_values(ascending=False, inplace=False)
    names = []
    for name, val in sum_values[0:5].iteritems():
        names.append(name)
    df3 = df2[names]
    return px.scatter_matrix(df3, height=800 , labels={"pov":"Poverty Rate", "h_income":"House Hold Income", "college":"College Rate", "urate":"Unemployment Rate", "share_black":"Black Population", "share_white":"White Population", "share_hispanic":"Hispanic Population", "age":"Age", "pop":"Population", "county_income":"County Income"})

def display_correlation():
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    correlated_data = df2.corr(method='pearson')
    # print(correlated_data)
    figure = go.Figure( data=go.Heatmap(
                   z=[correlated_data["age"],correlated_data["pop"], correlated_data["pov"], correlated_data["college"], correlated_data["urate"], correlated_data["county_income"], correlated_data["h_income"], correlated_data["share_white"], correlated_data["share_black"], correlated_data["share_hispanic"]],
                x=['Age', 'Population', 'Poverty Rate', 'College Rate', 'Unemployment Rate', 'County Income', 'House Hold Income', 'White Population', 'Black Population', 'Hispanic Population'], 
                   y=['Age', 'Population', 'Poverty Rate', 'College Rate', 'Unemployment Rate', 'County Income', 'House Hold Income', 'White Population', 'Black Population', 'Hispanic Population'],
            colorscale = 'RdBu', hoverongaps = False ))

    figure.update_layout(height = 500, 
    title='Correlation Matrix')
    return figure



if __name__ == '__main__':
    app.run_server(debug=True)

