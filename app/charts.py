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
app = dash.Dash()
with open("./original_police_Killings.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
df = pd.read_csv("./original_police_Killings.csv", encoding="ISO-8859-1")
df.head()
app.layout = html.Div(className='layout', children=[
    html.Nav(className="Navbar", children=[
    #     html.Ul(className="options", children=[
    #     html.Button("Correlation", id="correlationBtn"),
    #     html.Button("5×5 scatter plot matrix", id="scatter_plot_btn"),
    #    html.Button("parallel coordinates", id="parrallel_btn"),
    #    html.Button("PCA plot ", id="pca_btn"),
    #     html.Button("biplot", id="biplot_btn"),
    #     html.Button("MDS display(euclidian)", id="msd1_btn"),
    #     html.Button("MDS display(euclidian)", id="msd2_btn")
    #         ])
    dcc.Dropdown(id="options", options=[
            {'label': 'Correlation Matrix', 'value': 'correlation'},
            {'label': '5×5 scatter plot matrix', 'value': 'scatterplot'},
            {'label': 'parallel coordinates', 'value': 'parallel'},
            {'label': 'PCA plot', 'value': 'pca'},
            {'label': 'biplot', 'value': 'biplot'},
            {'label': 'State', 'value': 'State'},
            {'label': 'MDS display(euclidian)', 'value': 'msd1'},
            {'label': 'MDS display(euclidian)', 'value': 'msd2'},
           
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
    # elif options == 'msd1':
    #     return display_mds_euclidian(),  {'display':'none'},  dash.no_update
def display_mds_euclidian():
    figure = None
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    euc = euclidean_distances(df2)
    embedding = MDS(n_components=2, metric=True,  dissimilarity = "euclidean")
    df_transformed = embedding.fit_transform(df2)
    print(type(df_transformed))
    return figure
def display_biplot():
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    x = StandardScaler().fit_transform(df2)
    pca = PCA(n_components=10)
    components = pca.fit_transform(x)
    figure= px.scatter(components, x=0, y=1, color=df['raceethnicity'])
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
    y = pd.DataFrame(list(zip( eigenVals, indices,)), columns=[ "Eigen_Values","Index"])
    return px.bar(y, x='Index', y = 'Eigen_Values')
def display_pca():
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    df2.columns = ["Age", "Population","Poverty Rate","College rate", "Unemployment Rate", "County Income", "House hold income","White Population" , "Black Population" ,'Hispanic Population']
    features = ["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]
    x = StandardScaler().fit_transform(df2)
    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    figure= px.scatter(components, x=0, y=1, color=df['raceethnicity'])
    # df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    # df2.columns = ["Age", "Population","Poverty Rate","College rate", "Unemployment Rate", "County Income", "House hold income","White Population" , "Black Population" ,'Hispanic Population']
    # x = StandardScaler().fit_transform(df2)
    # pca = PCA(n_components=10)
    # components = pca.fit_transform(x)
    # indices = ['1', '2', '3', '4', '5', '6', '7',  '8',  '9',  '10']
    # eigenVals = []
    # for i in components.explained_variance:
    #     eigenVals.append(i)
    # dataF = pd.DataFrame(zip(eigenVals, indices), columns=["explained Variance", "eigen"])
    # figure= px.bar(dataF)
    return figure
def display_parallel():
    figure =None
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    correlated_data = df2.corr(method='pearson').abs().sum(axis=0).sort_values(ascending=False, inplace=False)
    A1 = correlated_data[0]
    correlated_data= df2.corr(method='pearson')
    # print(correlated_data)
    # already_viewed = ['pov']
    # for index in correlated_data.index:
    #     if(index != 'pov'):
    #         print(index)
    #         print(correlated_data[index].argmax())
    # print(already_viewed) 
    df3 = df2[["pov", "h_income", "college", "urate",'share_black', 'share_white',  'share_hispanic', 'age', 'pop', 'county_income']]
    figure = px.parallel_coordinates(df3 ,color_continuous_midpoint=2)
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
    return px.scatter_matrix(df3)

def display_correlation():
    df2 = df[["age","pop", "pov", "college", "urate", "county_income",  "h_income", "share_white", "share_black", "share_hispanic"]]
    correlated_data = df2.corr(method='pearson')
    # print(correlated_data)
    figure = go.Figure(data=go.Heatmap(
                   z=[correlated_data["age"],correlated_data["pop"], correlated_data["pov"], correlated_data["college"], correlated_data["urate"], correlated_data["county_income"], correlated_data["h_income"], correlated_data["share_white"], correlated_data["share_black"], correlated_data["share_hispanic"]],
                x=['Age', 'Population', 'Poverty Rate', 'College Rate', 'Unemployment Rate', 'County Income', 'House Hold Income', 'White Population', 'Black Population', 'Hispanic Population'], 
                   y=['Age', 'Population', 'Poverty Rate', 'College Rate', 'Unemployment Rate', 'County Income', 'House Hold Income', 'White Population', 'Black Population', 'Hispanic Population'],
            colorscale = 'RdBu', hoverongaps = False ))

    figure.update_layout(
    title='Correlation Matrix')
    return figure



if __name__ == '__main__':
    app.run_server(debug=True)

