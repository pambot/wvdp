import math
import os
import copy
import json
import numpy as np

from bokeh.io import show, output_notebook, output_file
from bokeh.plotting import figure
from bokeh.models import GraphRenderer, StaticLayoutProvider, HoverTool, TapTool, CustomJS
from bokeh.models.sources import ColumnDataSource
from bokeh.models.graphs import *
from bokeh.models.glyphs import *
from bokeh.models.arrow_heads import *
from bokeh.models.annotations import *
from bokeh.models.widgets import *
from bokeh.layouts import widgetbox
from bokeh.resources import CDN
from bokeh.embed import components

from flask import Flask, render_template


app = Flask(__name__)

RADIUS = 0.15
BACKGROUND = "#e0e0e0"
EDGE_LABELS = ["correlation", "undirected causal", "directed causal", "genuine causal"]

def make_figure(nodes, edges, correlation_edges, pos):
  
  # define gplot
  gplot = figure(title=None, x_range=(-2, 2), y_range=(-2, 2), tools="reset,save",
         plot_width=900, plot_height=900, match_aspect=False)

  graph = GraphRenderer()

  # add nodes
  light_node = Ellipse(height=RADIUS, width=RADIUS, fill_color="color", line_color="#000000", line_width=5)
  heavy_node = Ellipse(height=RADIUS, width=RADIUS, fill_color="color", line_color="#000000", line_width=7)
  
  for k in nodes:
    graph.node_renderer.data_source.add(nodes[k], k)
  
  graph.node_renderer.glyph = light_node
  graph.node_renderer.hover_glyph = heavy_node
  graph.node_renderer.selection_glyph = heavy_node
  graph.node_renderer.nonselection_glyph = light_node

  # add directed edges
  graph.edge_renderer.name = "edges"
  
  for k in correlation_edges:
    graph.edge_renderer.data_source.add(correlation_edges[k], k)

  light_edge = MultiLine(line_width="width", line_color="color", line_alpha="alpha")
  heavy_edge = MultiLine(line_width="width", line_color="color", line_alpha=1)
  
  graph.edge_renderer.glyph = light_edge
  graph.edge_renderer.hover_glyph = heavy_edge
  graph.edge_renderer.selection_glyph = light_edge
  graph.edge_renderer.nonselection_glyph = light_edge

  # arrows
  arrow = NormalHead(fill_color="#000000", line_color=None, size=8)
  arrow_source = ColumnDataSource(dict(x_start=[], y_start=[], x_end=[], y_end=[]))
  gplot.add_layout(
    Arrow(
      end=arrow, source=arrow_source,
      x_start="x_start", y_start="y_start", x_end="x_end", y_end="y_end"
    )
  )

  # add labels
  p_ind = np.linspace(0, 1-1/len(pos), len(pos)) * np.pi * 2
  xr = 1.1 * np.cos(p_ind)
  yr = 1.1 * np.sin(p_ind)
  rad = np.arctan2(yr, xr)
  gplot.text(xr, yr, nodes["index"], angle=rad,
    text_font_size="9pt", text_align="left", text_baseline="middle")

  # render
  graph.layout_provider = StaticLayoutProvider(graph_layout=pos)
  graph.inspection_policy = EdgesAndLinkedNodes()
  graph.selection_policy = NodesAndLinkedEdges()

  # widgets
  edges_original = ColumnDataSource(edges)

  slider = Slider(start=0.0, end=1.0, value=0.0, step=0.1, 
    title="absolute correlation coefficient is greater than")

  checkbox = CheckboxButtonGroup(
    labels=EDGE_LABELS, 
    active=[0]
  )

  callback = CustomJS(
    args=dict(
      graph=graph,
      edges_original=edges_original, 
      arrow_source=arrow_source,
      checkbox=checkbox,
      slider=slider
    ), 
    code="""
  	  var e = graph.edge_renderer.data_source.data;
  	  var n = graph.node_renderer.data_source.data;
      var a = arrow_source.data;
      var o = edges_original.data;
      var cb = checkbox.active;
      var sv = slider.value;
      var ns = graph.node_renderer.data_source.selected.indices;
      if (ns.length > 0) {
				var nn = n['index'][ns[0]];
				for (var key in o) {
					var vals = [];
					for (var i = 0; i < o['start'].length; ++i) {
						if ((o['start'][i] == nn || o['end'][i] == nn) && (Math.abs(o['r'][i]) > sv) && (cb.indexOf(o['type'][i]) > -1)) {
							vals.push(o[key][i]);
						}
					}
					e[key] = vals;
				}
				a['x_start'].length = 0;
				a['y_start'].length = 0;
				a['x_end'].length = 0;
				a['y_end'].length = 0;
				for (var i = 0; i < o['start'].length; ++i) {
					if ((o['start'][i] == nn || o['end'][i] == nn) && (Math.abs(o['r'][i]) > sv) && (cb.indexOf(o['type'][i]) > -1)) {
						if (o['e_arrow'][i] === 1) {
							var l = o['xs'][i].length;
							a['x_start'].push(o['xs'][i][l - 2]);
							a['y_start'].push(o['ys'][i][l - 2]);
							a['x_end'].push(o['xs'][i][l - 1]);
							a['y_end'].push(o['ys'][i][l - 1]);
						}
						if (o['b_arrow'][i] === 1) {
							a['x_start'].push(o['xs'][i][1]);
							a['y_start'].push(o['ys'][i][1]);
							a['x_end'].push(o['xs'][i][0]);
							a['y_end'].push(o['ys'][i][0]);
						}
					}
				}
			} else {
					for (var key in o) {
					var vals = [];
					for (var i = 0; i < o['start'].length; ++i) {
						if ((Math.abs(o['r'][i]) > sv) && (cb.indexOf(o['type'][i]) > -1)) {
							vals.push(o[key][i]);
						}
					}
					e[key] = vals;
				}
				a['x_start'].length = 0;
				a['y_start'].length = 0;
				a['x_end'].length = 0;
				a['y_end'].length = 0;
				for (var i = 0; i < o['start'].length; ++i) {
					if ((Math.abs(o['r'][i]) > sv) && (cb.indexOf(o['type'][i]) > -1)) {
						if (o['e_arrow'][i] === 1) {
							var l = o['xs'][i].length;
							a['x_start'].push(o['xs'][i][l - 2]);
							a['y_start'].push(o['ys'][i][l - 2]);
							a['x_end'].push(o['xs'][i][l - 1]);
							a['y_end'].push(o['ys'][i][l - 1]);
						}
						if (o['b_arrow'][i] === 1) {
							a['x_start'].push(o['xs'][i][1]);
							a['y_start'].push(o['ys'][i][1]);
							a['x_end'].push(o['xs'][i][0]);
							a['y_end'].push(o['ys'][i][0]);
						}
					}
				}
			}
      graph.edge_renderer.data_source.change.emit();
      arrow_source.change.emit();
  """)
  slider.js_on_change("value", callback)
  checkbox.js_on_change("active", callback)
  graph.node_renderer.data_source.selected.js_on_change("indices", callback)
  
  hover = HoverTool(
    tooltips=[
      ("node", "@start"),
      ("node", "@end"),
      ("type", "@type_name"),
      ("pearson", "@r{0.3f}"),
      ("p-value", "@pval"),
    ],
    renderers=[graph]
  )
  
  gplot.background_fill_color = BACKGROUND
  gplot.xgrid.grid_line_color = None
  gplot.ygrid.grid_line_color = None
  gplot.axis.visible = False
  gplot.add_tools(hover, TapTool())
  gplot.toolbar.logo = None
  gplot.toolbar_location = None
  gplot.border_fill_color = None
  gplot.outline_line_color = None
  gplot.renderers.append(graph)
  
  return {
    "gplot": gplot, 
    "widgets": widgetbox(slider, checkbox, width=450)
  }

@app.route("/wvdp/")
def chart():
  from data import nodes, edges, correlation_edges, pos
  
  f = make_figure(nodes, edges, correlation_edges, pos)
  script, (graph_div, widgets_div) = components([f["gplot"], f["widgets"]])
  return render_template("index.html", 
    graph_div=graph_div,
    widgets_div=widgets_div,
    script=script,
  )

if __name__ == "__main__":
  #app.run(host="0.0.0.0", port=80)
  app.run(debug=True)

