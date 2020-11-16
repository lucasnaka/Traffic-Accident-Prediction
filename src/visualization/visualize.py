import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.graph_objs import Figure, Data
import plotly.offline as pyoff
from ..utils.frame_utils import stratified_df

def plot_scatter_segment(df, xlabel, ylabel, segment_label='Segment', length=1000):
    """Generate a scatter plot, assigning a different colour to each unique value
    in `segment_label` using a stratified sample of length `length`.

    Args:
        df ([type]): [description]
        xlabel ([type]): [description]
        ylabel ([type]): [description]
        segment_label (str, optional): [description]. Defaults to 'Segment'.
        length (int, optional): [description]. Defaults to 1000.
    """    
    plot_data = []
    df_ = stratified_df(df, segment_label, length)
    for i, segment in enumerate(df_[segment_label].unique()):
        scatter = go.Scatter(
            x=df_.query(f"{segment_label} == '{segment}'")[xlabel],
            y=df_.query(f"{segment_label} == '{segment}'")[ylabel],
            mode='markers',
            name=segment,
            marker=dict(
                size=i+6,
                line=dict(width=1),
                autocolorscale=True,
                opacity=0.8
            )
        )
        plot_data.append(scatter)
    plot_layout = go.Layout(
        yaxis= {"title": f"{ylabel}"},
        xaxis= {"title": f"{xlabel}"},
        title=segment_label
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)
    
def pareto_gen(df, field):
    """Generate a Pareto graph, based on `field` cummulative proportion.

    Args:
        df ([type]): [description]
        field ([type]): [description]

    Returns:
        [type]: [description]
    """    

    df_pareto = round(df[field].value_counts(normalize=True), 2).to_frame()
    df_pareto['absolute'] = df[field].value_counts(normalize=False)
    df_pareto['categories'] = df_pareto.index.to_list()
    df_pareto['categories'] = df_pareto['categories']
    df_pareto.sort_values(by='absolute', ascending=False, inplace=True)
    #df_pareto.sort_values(by='categories', ascending=True, inplace=True)
    df_pareto['cum_sum'] = df_pareto['absolute'].cumsum()
    df_pareto['cum_perc'] = round(df_pareto.cum_sum/df_pareto['absolute'].sum(),2)
    df_pareto['pareto_80'] = 0.8
    _, labels = pd.cut(df_pareto[field], df_pareto.shape[0], retbins=True)
    max_label = np.max(labels)
    #labels = [str(round(i, 2))+" %" for i in labels]
    labels = labels[1:]
    idx = np.where(df_pareto['cum_perc'] >= 0.7999)[0][0]
    x_80, y_80 = df_pareto.iloc[idx, :][['categories', 'cum_perc']].values
    #y_80 = labels[-idx+2][:-2]
        
    trace1 = {
      "name": "Count", 
      "type": "bar", 
      "x": df_pareto['categories'].values, 
      "y": df_pareto[field].values, 
      "marker": {"color": "rgb(34,163,192)"}
    }
    trace2 = {
      "line": {
        "color": "rgb(243,158,115)", 
        "width": 2.4
      }, 
      "name": "Cumulative Percentage", 
      "type": "scatter", 
      "x": df_pareto['categories'].values, 
      "y": df_pareto['cum_perc'].values, 
      "yaxis": "y2"
    }
    trace3 = {
      "line": {
        "dash": "dash", 
        "color": "rgba(128,128,128,.45)", 
        "width": 1.5
      }, 
      "name": "80%", 
      "type": "scatter", 
      "x": df_pareto['categories'], 
      "y": df_pareto['pareto_80'], 
      "yaxis": "y2"
    }
    data = Data([trace1, trace2, trace3])
    layout = {
      "font": {
        "size": 12, 
        "color": "rgb(128,128,128)", 
        "family": "Balto, sans-serif"
      }, 
      "title": "Diagram Pareto ", 
      "width": 1500, 
      "xaxis": {"tickangle": -90,
                'type': 'category'
                }, 
      "yaxis": {
        "range": [0, max_label],
        "tickformat": ",.0%",
        "title": "Count", 
        "tickfont": {"color": "rgba(34,163,192,.75)"}, 
        "tickvals": labels, 
        "titlefont": {
          "size": 14, 
          "color": "rgba(34,163,192,.75)", 
          "family": "Balto, sans-serif"
        }
      }, 
      "height": 623, 
      "legend": {
        "x": 0.83, 
        "y": 1.3, 
        "font": {
          "size": 12, 
          "color": "rgba(128,128,128,.75)", 
          "family": "Balto, sans-serif"
        }
      }, 
      "margin": {
        "b": 250, 
        "l": 60, 
        "r": 60, 
        "t": 65
      }, 
      "yaxis2": {
        "side": "right", 
        "range": [0, 1.01],
        "tickformat": ',.0%',
        "tickfont": {"color": "rgba(243,158,115,.9)"}, 
        "tickvals": [0, 20, 40, 60, 80, 100], 
        "overlaying": "y"
      }, 
      "showlegend": True, 
      "annotations": [
        {
          "x": 1.029, 
          "y": 0.75, 
          "font": {
            "size": 14, 
            "color": "rgba(243,158,115,.9)", 
            "family": "Balto, sans-serif"
          }, 
          "text": "Cumulative Percentage", 
          "xref": "paper", 
          "yref": "paper", 
          "showarrow": False, 
          "textangle": 90
        },
        {
            "x": x_80,
            "y": y_80,
            "xref": "x",
            "yref": "y2",
            "text": ">= 80 % of ammount",
            "showarrow": True,
            "font": {
                "size": 16, 
                "color": "rgba(243,158,115,.9)", 
                "family": "Balto, sans-serif"
            },
            "arrowhead": 2,
            "ax": 16,
            "ay": -60
        }
      ], 
      "plot_bgcolor": "rgb(240, 240, 240)", 
      "paper_bgcolor": "rgb(240, 240, 240)"
    }
    
    return Figure(data=data, layout=layout)
