#!/usr/bin/env python
# encoding: utf-8

import plotly.express as px

from plotly.subplots import make_subplots
import plotly.graph_objects as go

"""
PX

Klasse mit Methoden zur Erstellung von Diagrammen mit plotly
"""
class PX(object):
    #def __init__(self):
    
    """
    Histogramm
    """
    @staticmethod
    def hist(values, xaxes = {'title': 'x'}, yaxes = {'title': 'y'}, width=900):
        fig = px.histogram(values)
    
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', title=yaxes['title'])
        fig.update_xaxes(showline=True, linewidth=1, linecolor='LightGrey', title=xaxes['title'])
        
        fig.update_layout(width=width,
                          plot_bgcolor='rgba(255, 255, 255, 1.0)', 
                          paper_bgcolor='rgba(255, 255, 255, 1.0)', 
                          #margin={'l':0, 'b':0, 'r':0, 't':0},
                          showlegend=False)#
    
        fig.show()
        
    """
    Bar-Plot
    """
    @staticmethod
    def bar(values, xaxes = {'title': 'x'}, yaxes = {'title': 'y'}, x='x', y='y', width=900, color=None, barmode='relative', category_orders=None):
        fig = px.bar(values, x=x, y=y, color=color, barmode=barmode, category_orders=category_orders)
    
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', title=yaxes['title'])
        fig.update_xaxes(showline=True, linewidth=1, linecolor='LightGrey', title=xaxes['title'])
        
        fig.update_layout(width=width,
                          plot_bgcolor='rgba(255, 255, 255, 1.0)', 
                          paper_bgcolor='rgba(255, 255, 255, 1.0)', 
                          #margin={'l':0, 'b':0, 'r':0, 't':0},
                          showlegend=True)#
        
        fig.show()

    """
    Box-Plot
    """
    @staticmethod
    def box(values, xaxes = {'title': 'x'}, yaxes = {'title': 'y'}, width=900):
        fig = px.box(values, points='outliers', notched=True)#violin#all
    
        fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='LightGrey', title=yaxes['title'])
        fig.update_xaxes(showline=True, linewidth=1, linecolor='LightGrey', title=xaxes['title'])
        
        fig.update_layout(width=width,
                          plot_bgcolor='rgba(255, 255, 255, 1.0)', 
                          paper_bgcolor='rgba(255, 255, 255, 1.0)', 
                          #margin={'l':0, 'b':0, 'r':0, 't':0},
                          showlegend=False)#
        
        fig.show()

    """
    Multi plots
    """

    """
    Histogramm
    """
    def hist_multi(df, unique, values, xaxes={'title':''}, yaxes={'title':''}, title='', width=900, height=1600):
        df_unique = df[unique].unique()
        fig = make_subplots(rows=df_unique.shape[0], cols=1)

        #Histogramm für alle eindeutigen Werte der Spalte aus der Variablen 'unique' zu Plot hinzufügen
        for i in range(0,len(df_unique)):
            entry = df_unique[i]
            fig.add_trace(
                go.Histogram(x=df.loc[df[unique] == entry, values], name=entry, xbins={'size': 1}),#
                row=i + 1, col=1
            )

        # Festlegung des unteren und oberen Bereichs für x- und y-Achse nach Maximalwert
        fig.update_xaxes(showline=True, linewidth=1, linecolor='LightGrey', range=[0, df[values].max()], title=xaxes['title'])
        #fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', title='count')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', range=[0, df.groupby(by=[unique])[values].value_counts().max()], title=yaxes['title'])
        
        fig.update_layout(
            title_text=title,
            width=width,
            height=height, 
            plot_bgcolor='rgba(255, 255, 255, 1.0)', 
            paper_bgcolor='rgba(255, 255, 255, 1.0)', 
            #margin={'l':0, 'b':0, 'r':0, 't':0},
            #showlegend=False
        )
        
        fig.show()

    """
    Box-Plot 
    """
    def box_multi(df, unique, values, xaxes={'title':''}, yaxes={'title':''}, width=900):
        fig = go.Figure()

        #Box-Plot für alle eindeutigen Werte der Spalte aus der Variablen 'unique' zu Plot hinzufügen
        for entry in df[unique].unique():
            fig.add_trace(go.Box(y=df.loc[df[unique] == entry, values], name=entry, boxpoints="all"))
        
        fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='LightGrey', title=yaxes['title'])
        fig.update_xaxes(showline=True, linewidth=1, linecolor='LightGrey', title=xaxes['title'])
        
        fig.update_layout(width=width,
            plot_bgcolor='rgba(255, 255, 255, 1.0)', 
            paper_bgcolor='rgba(255, 255, 255, 1.0)', 
            #margin={'l':0, 'b':0, 'r':0, 't':0},
                      showlegend=False)#
        
        fig.show()

    """
    Scatter-Line
    """
    def scatter_line_multi(df, unique, x, y, title, xaxes={'title':''}, yaxes={'range':[0.0, 0.1], 'title':''}, width=900):
        fig = go.Figure()

        # Scatter-Line-Plot für alle eindeutigen Werte der Spalte aus der Variablen 'unique' zu Plot hinzufügen
        for entry in df[unique].unique():
            fig.add_trace(go.Scatter(x=df.loc[df[unique] == entry, x], 
                y=df.loc[df[unique] == entry, y], 
                name=entry,
                mode='lines+markers')
            )
        
        fig.update_yaxes(showline=True, linewidth=1, linecolor='LightGrey', showgrid=False, range=yaxes['range'], title=yaxes['title'] if len(yaxes['title']) > 0 else y)
        fig.update_xaxes(showline=True, linewidth=1, linecolor='LightGrey', showgrid=False, title=xaxes['title'] if len(xaxes['title']) > 0 else x)#, dtick = 0.1
        
        fig.update_layout(width=width,
            plot_bgcolor='rgba(255, 255, 255, 1.0)', 
            paper_bgcolor='rgba(255, 255, 255, 1.0)', 
            #margin={'l':0, 'b':0, 'r':0, 't':0},
            title=title,
            showlegend=True)#
        
        fig.show()
    
    """
    Confusion-Matrix
    """
    def displayConfusionMatrix(cm, xaxes={'labels': [], 'title': ''}, yaxes={'labels': [], 'title': ''}, width=None, height=None, ticktextlen=None):
        fig = px.imshow(
            cm, 
            text_auto=True, 
            width=width, 
            height=height,
            color_continuous_scale  = 'Blues'
        )
        
        fig.update_xaxes(
            tickvals=[*range(len(xaxes['labels']))], 
            # Label der Klassen und relative Häufigkeit für Achse bei binärer Klassifikation
            # sonst Label der Klassen mit optionaler Kürzung der Texte
            ticktext=[f'{xaxes["labels"][i]} ({cm[i][i]/cm.sum(axis=0)[i]:.2f})' for i in range(len(xaxes['labels']))] if len(xaxes['labels']) == 2 else (xaxes['labels'] if ticktextlen == None else [f'{s[:ticktextlen]}...' for s in xaxes['labels']]),
            tickmode='array', 
            title=xaxes['title'])
        
        fig.update_yaxes(
            tickvals=[*range(len(yaxes['labels']))], 
            # Label der Klassen und relative Häufigkeit für Achse bei binärer Klassifikation
            # sonst Label der Klassen mit optionaler Kürzung der Texte
            ticktext=[f'{yaxes["labels"][i]} ({cm[i][i]/cm.sum(axis=1)[i]:.2f})' for i in range(len(yaxes['labels']))] if len(yaxes['labels']) == 2 else (yaxes['labels'] if ticktextlen == None else [f'{s[:ticktextlen]}...' for s in yaxes['labels']]),
            tickmode='array', 
            title=yaxes['title'])
    
        fig.show()