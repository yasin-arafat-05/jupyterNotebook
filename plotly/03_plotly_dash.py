
""" 
- Installation of Dash:
    - 
    - pip install dash  
    -
    
=========================== Design this ===========================
+-------------------------------------------------------------+
|                           HEADER                            |
+-------------------------------------------------------------+
|  +----------------------------+----------------------------+ |
|  |                            |                            | |
|  |          LEFT AREA         |         RIGHT AREA         | |
|  |                            |                            | |
|  +----------------------------+----------------------------+ |
+-------------------------------------------------------------+
"""

import dash
import pandas as pd 
from dash import dcc
from dash import html 
import plotly.express as px


app = dash.Dash()

# graph:
# Sample data
df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [10, 20, 15, 30, 25],
    "category": ["A", "A", "B", "B", "A"]
})

# Create scatter plot
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="category",
    title="My Scatter Plot"
)

fig.update_layout(
    xaxis={"title":"x-values"},
    yaxis={"title":"y-values"},
)

# define the layout:
app.layout = html.Div([ # first div for all 

    html.Div(children=[
            html.H1("My first dashboard.",style={"color":'red','text-align':'center'}),
        ],
    style={"border":"1px black solid","width":'100%','height':'100px','float':'left'}
    ),
    html.Div(style={'border':'1px black solid','width':'49.85%','height':'350px','float':'left'}
    ),
    html.Div([
         dcc.Graph(
            id="scatter",
            figure=fig
        )
        
    ],
    style={'border':'1px black solid','width':'49.85%','height':'350px','float':'left'}
    )
])


if __name__ == '__main__':
    app.run(debug=True,use_reloader=True)

