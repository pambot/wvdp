import math
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex

from bokeh.io import show, output_notebook, output_file
from bokeh.plotting import figure
from bokeh.models import GraphRenderer, StaticLayoutProvider, HoverTool, TapTool, CustomJS
from bokeh.models.sources import ColumnDataSource
from bokeh.models.graphs import NodesAndLinkedEdges
from bokeh.models.glyphs import *
from bokeh.models.arrow_heads import *
from bokeh.models.annotations import *
from bokeh.layouts import gridplot
from bokeh.resources import CDN
from bokeh.embed import components

from networkx.readwrite.json_graph import node_link_graph
import pickle

from flask import Flask, render_template

app = Flask(__name__)

RADIUS = 0.15
STEPS = 1000
OUTLINE = "#383838"
BACKGROUND = "#e0e0e0"

def dist(l1, l2):
    x1, y1 = l1
    x2, y2 = l2
    return ((y2 - y1)**2 + (x2 - x1)**2)**0.5

def bezier(l1, l2, b):
    x1, y1 = l1
    x2, y2 = l2
    d = dist(l1, l2)
    t = b * (1 + d)
    steps = [i/STEPS for i in range(STEPS)]
    xs = [(1-s)**t*x1 + s**t*x2 for s in steps]
    ys = [(1-s)**t*y1 + s**t*y2 for s in steps]
    return xs, ys

def colormap(num):
    colors = ["#f7c031", "#ef4837", "#91b5bb", "#526354", "#fecacb"]
    return list(colors * 100)[:num]

def nearest_offset(xs, ys, centroid):
    for i, (x, y) in enumerate(zip(xs[::-1], ys[::-1])):
        if dist(centroid, (x, y)) > RADIUS-0.075:
            break
    return i

def make_figure(D, pos, df):
		data = ColumnDataSource(df)
		
		# define gplot
		gplot = figure(title=None, x_range=(-2, 2), y_range=(-2, 2),
									tools="reset,save", plot_width=850, plot_height=850, match_aspect=False)
		gplot.background_fill_color = BACKGROUND
		gplot.xgrid.grid_line_color = None
		gplot.ygrid.grid_line_color = None
		gplot.axis.visible = False
		gplot.add_tools(HoverTool(tooltips=None))
		gplot.toolbar.logo = None
		gplot.toolbar_location = None
		gplot.border_fill_color = None
		gplot.outline_line_color = None

		graph = GraphRenderer()

		# add nodes
		light_node = Ellipse(height=RADIUS, width=RADIUS, fill_color="color", line_color=OUTLINE, line_width=5)
		heavy_node = Ellipse(height=RADIUS, width=RADIUS, fill_color="color", line_color=OUTLINE, line_width=7)
		
		graph.node_renderer.data_source.add(list(pos.keys()), "index")
		graph.node_renderer.data_source.add(list(pos.keys()), "label")
		graph.node_renderer.data_source.add(colormap(len(pos)), "color")
		graph.node_renderer.data_source.add([pos[n][0] for n in pos.keys()], "x")
		graph.node_renderer.data_source.add([pos[n][1] for n in pos.keys()], "y")
		graph.node_renderer.glyph = light_node
		graph.node_renderer.hover_glyph = heavy_node

		# add directed edges
		graph.edge_renderer.data_source.data = dict(start=[], end=[], xs=[], ys=[], edge_color=[])

		for e in D.edges():
				n1, n2 = e
				l1, l2 = pos[n1], pos[n2]
				xs, ys = bezier(l1, l2, 1)
				os = nearest_offset(xs, ys, l2)
				graph.edge_renderer.data_source.data["start"].append(n1)
				graph.edge_renderer.data_source.data["end"].append(n2)
				graph.edge_renderer.data_source.data["xs"].append(xs[os:-os])
				graph.edge_renderer.data_source.data["ys"].append(ys[os:-os])
				graph.edge_renderer.data_source.data["edge_color"].append(OUTLINE)

		light_edge = MultiLine(line_width=2, line_color="edge_color", line_alpha=0.5)
		heavy_edge = MultiLine(line_width=5, line_color="edge_color", line_alpha=1)
		
		graph.edge_renderer.glyph = light_edge
		graph.edge_renderer.hover_glyph = heavy_edge
		graph.edge_renderer.selection_glyph = light_edge
		graph.edge_renderer.nonselection_glyph = light_edge
		
		# 		for e, xs, ys in zip(D.edges(data=True),
		# 												 graph.edge_renderer.data_source.data["xs"],
		# 												 graph.edge_renderer.data_source.data["ys"]):
		# 				n1, n2, data = e
		# 				l1, l2 = pos[n1], pos[n2]
		# 				x1, y1 = l1
		# 				x2, y2 = l2
		# 				if data["directed"]:
		# 						os = nearest_offset(xs, ys, l2)
		# 						arrow = VeeHead(fill_color=OUTLINE, line_color=None, size=20)
		# 						gplot.add_layout(
		# 								Arrow(end=arrow, x_start=xs[-os-1], y_start=ys[-os-1], x_end=xs[-os], y_end=ys[-os], line_color=None)
		# 						)
		# 						if data["both"]:
		# 								gplot.add_layout(
		# 										Arrow(end=arrow, x_start=xs[os+1], y_start=ys[os+1], x_end=xs[os], y_end=ys[os], line_color=None)
		# 								)

		# add labels
		p_ind = np.linspace(0, 1-1/len(pos), len(pos)) * np.pi * 2
		xr = 1.1 * np.cos(p_ind)
		yr = 1.1 * np.sin(p_ind)
		rad = np.arctan2(yr, xr)
		gplot.text(xr, yr, list(D.nodes()), angle=rad,
				text_font_size="9pt", text_align="left", text_baseline="middle")

		# render
		graph.layout_provider = StaticLayoutProvider(graph_layout=pos)
		graph.inspection_policy = NodesAndLinkedEdges()
		
		splot = figure(title=None, tools="pan,zoom_in,zoom_out,box_zoom,reset,save", plot_width=700, plot_height=850)

		shover = HoverTool(tooltips=[
				("country", "@country")
		])

		splot.background_fill_color = BACKGROUND
		splot.add_tools(shover)
		splot.toolbar.logo = None
		splot.toolbar_location = "left"
		splot.border_fill_color = None
		splot.outline_line_color = None
		
		sx = "tax burden score"
		sy = "GINI index"
		data.add(df[sx], "x")
		data.add(df[sy], "y")
		
		splot.circle(x="x", y="y", 
				source=data, radius="radius", color="#526354"
		)
		splot.xaxis.axis_label = sx
		splot.yaxis.axis_label = sy

		# 		gplot.js_on_event('tap', CustomJS(args=dict(data=data), code="""
		# 				label = cb_obj.label;
		# 				data['x'] = data[label];
		# 				data.change.emit()
		# 		"""))
		
		gplot.renderers.append(graph)
		
		return gplot, splot

def process_data(df):
		df.drop("ISO Country code", axis=1, inplace=True)
		df.dropna(axis=1, how="all", inplace=True)
		df.loc[:, "radius"] = (df["population"]/df["population"].max() + 0.03)*5
		
		return df

@app.route("/wvdp/")
def chart():
		with open("data/graph", "rb") as f:
				D = node_link_graph(pickle.load(f))

		with open("data/pos", "rb") as f:
				pos = pickle.load(f)

		df = pd.read_csv("data/wdvp_stats.tsv", 
                 sep="\t", 
                 header=0, 
                 skiprows=range(1, 5),
                 thousands=',',
                 na_values=["-"])
		df = process_data(df)
		gplot, splot = make_figure(D, pos, df)
		graph_script, graph_div = components(gplot)
		scatter_script, scatter_div = components(splot)
		return render_template("index.html", 
				graph_div=graph_div,
				graph_script=graph_script,
				scatter_script=scatter_script,
				scatter_div=scatter_div
		)

if __name__ == "__main__":
		app.run(debug=True)

