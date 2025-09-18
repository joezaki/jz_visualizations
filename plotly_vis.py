import os
import warnings
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from numpy.polynomial.polynomial import polyfit
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from matplotlib.colors import to_rgb


# -----------------------------------------

def compute_agg_data(
    data,
    group_var,
    sep_var,
    overlay_var,
    plot_var,
    central_tendency='mean',
    error_type='sem',
    observed=False
    ):
    '''
    Takes in an input dataframe and aggregates it using functions for central tendency and error.
    Returns a dataframe for central tendency, one for error, and associated xlabels.

    Parameters
    ==========
    data : pandas dataframe
        aggregated input dataframe in long format
    group_var : str
        name of the column in data by which data will be grouped.
    sep_var, overlay_var : str
        names of the columns for which there should be one unique value. Column will be aggregated by this one unique
        value, or if there is more than one unique value, will be changed to np.nan out of caution.
    plot_var : str
        name of the column in data representing the readout of interest to aggregate.
    central_tendency : str
        which measure of central tendency to use when aggregating data. Default is 'mean'.
    error_type : str
        which measure of error to use when aggregating data. Default is 'sem'.
    observed : bool
        whether or not to include missing data in categorical variables. Default is False.
    '''
    grouped_data = data[[sep_var, overlay_var, group_var, plot_var]].groupby(group_var, observed=observed)

    unique_val_func = lambda x: x.unique()[0] if x.nunique() == 1 else np.nan
    agg_data = grouped_data.agg({sep_var:unique_val_func,
                                        overlay_var:unique_val_func,
                                        plot_var:central_tendency}).sort_index()
    error_data   = grouped_data.agg({sep_var:unique_val_func,
                                        overlay_var:unique_val_func,
                                        plot_var:error_type}).sort_index()
    
    xlabels = agg_data.index.values
    agg_data   = agg_data.reset_index().rename(columns={'index': group_var})
    error_data = error_data.reset_index().rename(columns={'index': group_var})
    agg_data.index = xlabels
    error_data.index = xlabels

    return agg_data, error_data, xlabels


# -----------------------------------------


def agg_plot(
    data,
    plot_var,
    color_var=None,
    sep_var=None,
    group_var=None,
    overlay_var=None,
    central_tendency='mean',
    error_type='sem',
    datapoint_var='Subject',
    plot_mode='bar',
    plot_agg=True,
    bar_mode='group',
    plot_datapoints=False,
    plot_datalines=False,
    agg_marker_size=15,
    agg_marker_shape='',
    agg_line_width=3,
    error_width=3,
    colors='slategrey',
    color_datapoints=False,
    opacity=1,
    title=None,
    x_title=None,
    y_title=None,
    y_range=None,
    x_dtick=None,
    y_dtick=None,
    match_y_ranges=True,
    add_hline=True,
    hline_y=0,
    text_size=18,
    font_family='Arial',
    plot_width=600,
    plot_height=600,
    tick_angle=45,
    h_spacing=0.1,
    shapes_to_add=None,
    save_path=None,
    plot_scale=5
    ):
    '''
    A general purpose plotting function to plot summary data in bar, point, or line graph form, separating
    across up to three variables. Returns an interactive plotly graph, and optionally saves the graph in static
    or interactive format.

    If group_var, sep_var, and overlay_var are not of dtype None or pd.Categorical, they will be coerced to dtype
    pd.Categorical. At least one of these variables must not be None.

    For coloring, color_var must be one of sep_var, group_var, or overlay_var, or None. If one of those variables,
    colors must be a dict with a key-value pair for each unique entry and an associated color, or a string to uniformly
    color all the datapoints. If color_var is None, colors must be a string.

    Parameters
    ==========
    data : pandas dataframe
        aggregated input dataframe in long format
    plot_var : str
        name of the column in data representing the readout of interest
    color_var : str
        name of the column by which to color code the data. Must be of type pd.Categorical.
    group_var : str
        name of the column in data by which to group within a subplot. Must be of type pd.Categorical. Default is None.
    sep_var : str
        name of the column in data by which to separate across subplots. Must be of type pd.Categorical. Default is None.
    overlay_var : str
        name of the column in data by which to overlay within a subplot. Must be of type pd.Categorical. Default is None.
    central_tendency : str
        which measure of central tendency to use when aggregating data. Default is 'mean'.
    error_type : str
        which measure of error to use when aggregating data. Default is 'sem'.
    datapoint_var : str
        name of the column in data representing each individual subject name. Default is 'Subject'.
    plot_mode : str
        one of 'bar', 'line', or 'point' for which type of plot is desired. Default is 'bar'.
    plot_agg : bool
        whether or not to plot the aggregate data. Default is True.
    bar_mode : str
        how to group bars along the overlay_var variable. One of 'group' or 'overlay'. Only used if plot_mode=='bar'. Default is 'group'.
    plot_datapoints, plot_datalines : bool
        whether or not to plot individual subject datapoints of datalines. Defaults are False.
    agg_marker_size : int or float
        size of the datapoint representing the aggregated data of each group. Only used if plot_mode=='point'. Default is 15.
    agg_marker_shape : str
        pattern of bars in each bar plot. Only used if plot_mode=='bar'. Default is ''.
    agg_line_width : int or float
        width of the line of the aggregate line. Only used if plot_mode=='line'. Default is 3.
    error_width : int or float
        width or error bars. Default is 3.
    colors : dict, list/array, or str
        dictionary where each key is each unique color_var type and each value is its corresponding color, or a list
        where each entry represents a color for each unique color_var type (which will be inferred automatically), or a
        string representing one color for all datapoints. If color_var is None, must be a string. Default is 'slategrey'.
    color_datapoints : bool
        whether or not to individually color individual subjects' datapoints. Otherwise, all individual datapoints are
        black. Only used if plot_datapoints or plot_datalines==True. Default is False.
    opacity : int or float
        number representing opacity of aggregated data. Must range between [0,1]. Default is 1.
    title : str
        master title of the entire plot. Default is None.
    x_title, y_title : str
        title of the y-axis. Default is None.
    y_range : tuple
        tuple representing min and max y-axis values of plot range. If None, it is the range of the input data. Default is None.
    match_y_ranges : bool
        whether or not to match the y-axis ranges across subplots. Default is True.
    add_hline: bool
        whether or not to add a solid black line at a specified y-axis value. Default is True.
    hline_y : int or float
        y-axis value at which to draw a solid black line. Only used if add_hline==True. Default is 0.
    text_size : int or float
        size of all the text in the plot. Default is 18.
    font_family : str
        font family used in the plot. Default is 'Arial'.
    plot_width, plot_height : int or float
        width and height of the plot. Defaults are 600 and 600.
    tick_angle : int
        angle at which x-axis label text is displayed. Default is 45.
    h_spacing : float
        spacing between subplots. Only used if sep_var is not None. Default is 0.1.
    shapes_to_add : dict or list of dicts
        shape to be added to plot. Must be either a dict or a list of dicts of plotly shapes to be added. Default is None.
    save_path : str
        file path location including filename where plot should be saved. If None, plot will not be saved. Default is None.
    plot_scale : int
        size scaling of the plot. Only used if save_path is not None and if save_path extension is of a static type. Default is 5.
    '''

    vars_dict = {
        'sep_var'    : sep_var,
        'group_var'  : group_var,
        'overlay_var': overlay_var
        }

    # add placeholder for columns that are not specified
    for var_key, var in vars_dict.items():
        if var is None:
            data[var_key] = ''
            data[var_key] = pd.Categorical(data[var_key])
            vars_dict[var_key] = var_key
        else:
            # cast any non-categorical var to categorical
            if data[var].dtype is not pd.Categorical:
                data[var] = pd.Categorical(data[var])

    # configure colors
    if not (color_var in [sep_var, group_var, overlay_var]) | (color_var is None):
        raise Exception("Invalid color_var, must be equal to sep_var, group_var, overlay_var, or None.")
    if (color_var is None) & (type(colors) != str):
        raise Exception("If color_var is None, colors must be a string.")
    if (color_var is not None) & (type(colors) == str):
        colors = {unique_val:colors for unique_val in data[color_var].unique()}     # set uniform color if color is a string
    if (color_var is not None) & ((type(colors) == list) | (type(colors) == np.array)):
        colors = {unique_val:color for color, unique_val in zip(colors, data[color_var].unique().sort_values())} # infer colors dict by their order if colors is list-like

    # initialize plot
    subplot_titles = data[vars_dict['sep_var']].unique().sort_values()
    fig = make_subplots(rows=1, cols=len(subplot_titles), subplot_titles=subplot_titles,
                        horizontal_spacing=h_spacing, shared_yaxes=match_y_ranges)
    
    # separate data by variables and plot
    for i, sep in enumerate(data[vars_dict['sep_var']].unique().sort_values()):
        sep_data = data[data[vars_dict['sep_var']] == sep]
        for overlay in sep_data[vars_dict['overlay_var']].unique().sort_values():
            overlay_data = sep_data[sep_data[vars_dict['overlay_var']] == overlay]
            agg_data, error_data, xlabels = compute_agg_data(
                data             = overlay_data,
                sep_var          = vars_dict['sep_var'],
                overlay_var      = vars_dict['overlay_var'],
                group_var        = vars_dict['group_var'],
                plot_var         = plot_var,
                central_tendency = central_tendency,
                error_type       = error_type
                )
            agg_colors = [colors[c] if c is not np.nan else 'black' for c in agg_data[color_var].values] if color_var is not None else \
                          np.repeat(colors, agg_data.shape[0])

            if plot_agg:
                if plot_mode.lower() == 'bar':
                    fig.add_trace(
                        go.Bar(
                            x=xlabels,
                            y=agg_data[plot_var][xlabels].values,
                            error_y=dict(type='data', array=error_data[plot_var][xlabels].values, visible=True, width=error_width),
                            name=overlay,
                            marker=dict(color=agg_colors, line=dict(width=1, color='black'), opacity=opacity),
                            marker_pattern_shape=agg_marker_shape
                        ),
                        row=1,
                        col=i+1
                    )
                elif plot_mode.lower() == 'point':
                    fig.add_trace(
                        go.Scattergl(
                            x=xlabels,
                            y=agg_data[plot_var][xlabels].values,
                            error_y=dict(type='data', array=error_data[plot_var][xlabels].values, visible=True, width=error_width),
                            name=overlay,
                            mode='markers',
                            marker=dict(color=agg_colors, size=agg_marker_size, line=dict(width=1, color='black'), opacity=opacity),
                        ),
                        row=1,
                        col=i+1
                    )
                elif plot_mode.lower() == 'line':
                    fig.add_trace(
                        go.Scattergl(
                            x=xlabels,
                            y=agg_data[plot_var][xlabels].values,
                            error_y=dict(type='data', array=error_data[plot_var][xlabels].values, visible=True, width=error_width),
                            name=overlay,
                            mode='lines+markers',
                            marker=dict(color=agg_colors, size=agg_marker_size),
                            line=dict(color=agg_colors[0], width=agg_line_width),
                        ),
                        row=1,
                        col=i+1
                    )
                else:
                    raise Exception("Invalid plot_mode. Must be one of 'bar', 'point', or 'line'.")
        
            # plot individual datapoints
            if plot_datapoints | plot_datalines:
                datapoint_plot_mode = 'lines+markers' if (plot_datapoints & plot_datalines) else 'lines' if plot_datalines else 'markers'
                for point in overlay_data[datapoint_var].unique():
                    point_data = overlay_data[overlay_data[datapoint_var] == point].sort_values(vars_dict['group_var'])

                    # add warning if datapoint has more than 1 unique data entry for the vars_dict columns
                    if np.any(point_data[list(vars_dict.values())].value_counts() > 1):
                        warnings.warn('Datapoint {p} is represented more than once.'.format(p=point))
                    
                    point_color = [colors[c] for c in point_data[color_var].values] if color_var is not None and color_datapoints else \
                                   np.repeat('slategrey', point_data.shape[0])
                    line_color = point_color[0]
                    fig.add_trace(
                        go.Scattergl(
                            x=point_data[vars_dict['group_var']].values,
                            y=point_data[plot_var].values,
                            mode=datapoint_plot_mode,
                            marker=dict(color=point_color, symbol='circle-open', opacity=0.8, size=10),
                            line=dict(width=1, color=line_color),
                            name=str(point),
                        ),
                        row=1,
                        col=i+1
                    )

    # configure plot
    if add_hline:
        fig.add_hline(y=hline_y, row=1, col='all', line_width=1, opacity=1, line_color='black')
    fig.update_layout(
        dragmode="pan",
        font=dict(size=text_size, family=font_family),
        title_text=title,
        title_x=0.5,
        yaxis_title=y_title,
        autosize=False,
        width=plot_width,
        height=plot_height,
        template="simple_white",
        showlegend=False,
        barmode=bar_mode
    )
    fig.update_xaxes(tickangle=tick_angle, title_text=x_title, dtick=x_dtick, matches='x')
    fig.update_yaxes(range=y_range, dtick=y_dtick)

    # add shapes
    if shapes_to_add is not None:
        if type(shapes_to_add) == dict:
            shapes_to_add = [shapes_to_add] # if dict, turn to list of one element
        if type(shapes_to_add) == list:
            for shape in shapes_to_add:
                for col in np.arange(len(subplot_titles)):
                    fig.add_shape(shape, row=1, col=col+1)
        else:
            raise Exception("Invalid argument type 'shapes_to_add'. Must be one of dict or list.")

    # save plot
    if save_path is not None:
        save_dirname = os.path.dirname(save_path)
        if not os.path.exists(save_dirname):
            os.makedirs(save_dirname)
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(config=config)


# -----------------------------------------


def correlation_plot(
    data_x,
    data_y,
    groups=None,
    x_title=None,
    y_title=None,
    colors='slategrey',
    color_palette=px.colors.qualitative.Safe,
    corr_method='pearson',
    title="",
    textinfo=None,
    plot_points=True,
    plot_fits=True,
    plot_ci=False,
    plot_identity=False,
    same_xy_scale=False,
    showlegend=True,
    x_range=None,
    y_range=None,
    text_size=18,
    font_family='Arial',
    marker_size=7,
    outline_width=1,
    line_width=2,
    opacity=0.8,
    plot_height=600,
    plot_width=600,
    save_path=None,
    plot_scale=5
    ):
    """
    Plots correlation between two input vectors (data_x and data_y). Given an optional associated vector
    called groups, 
    Given two sorted vectors ('data_x' & 'data_y') and an optional associated 'groups' vector, plot the
    correlation between them, coloring each group separately.

    Parameters:
    ==========
    data_x, data_y : numpy 1d vector or list
        data values to be plotted along x-axis and y-axis respectively
    groups : numpy 1d vector or list
        the group identity that each datapoint belongs to. Must be of same length as data_x and data_y, or None. Default is None.
    x_title, y_title : str
        title text for x-axis and y-axis respectively. Default is None.
    colors : None or str or vector/list of colors
        colors to plot the datapoints. If None, color_palette will be discretized based on the number of
        unique values in groups. Each datapoint will be assigned a color based on the group it belongs to.
        If a list/vector of colors, length must be equal to length of data and the colors should match the
        group that each datapoint belongs to. If a string, all datapoints will be that color. Default is 'slategrey'.
    color_palette : plotly express color palette
        a color palette which is only used if colors = None (see above argument, colors). Default is px.colors.qualitative.Safe.
    corr_method : str
        which type of correlation test to perform. One of 'pearson', 'spearman', 'kendall'. Default is 'pearson'.
    title : str
        title for overall plot. Note that the title will also include the r and p values for
        line of best fit following the text provided. Default is ''.
    textinfo : None or 1d vector or list
        the text that should appear over each datapoint during hover. If not None, must be of same length as data. Default is None.
    plot_points : boolean
        whether or not to plot the individual datapoints as a scatterplot. Default is True.
    plot_fits : boolean
        whether or not to plot the lines of best fit for each unique group type. If False, only datapoints will be plotted.
        Default is True.
    plot_ci : boolean
        whether or not to plot the confidence interval around the line(s) of best fit. Default is False.
    plot_identity : boolean
        whether or not to plot the identity line as a black dotted line. Default is False.
    same_xy_scale : boolean
        whether to set the ranges of the x- and y-axes to the same ranges. Default is False.
    showlegend : boolean
        whether to show the legend designating what each line represents. Default is True.
    x_range, y_range : None or tuple
        the range to set the x-axis and y-axis, respectively. If both are of type tuple and same_xy_scale
        is set to False, the plot will be updated with the specified axis ranges. Defaults are None.
    text_size : int or float
        size of all the text in the plot. Default is 18.
    font_family : str
        font family used in the plot. Default is 'Arial'.
    marker_size : float
        size of each datapoint. Default is 7.
    outline_width : float
        width of the outlines of each datapoint. Default is 1.
    line_width : float
        width of the line of best fit. Only used if plot_fits is True. Default is 2.
    opacity : float
        how opaque each datapoint should be, from [0,1]. Default is 0.8.
    plot_height, plot_width : int
        the height and width, respectively, of the entier plot. Defaults are 600 and 600, respectively.
    save_path : str
        file path location including filename where plot should be saved. If None, plot will not be saved. Default is None.
    plot_scale : int
        size scaling of the plot. Only used if save_path is not None and if save_path extension is of a static type. Default is 5.
    """

    # prepare data
    groups = np.array(groups) if groups is not None else np.repeat(' ', len(data_x))

    # if no color is set, use a color palette to color each group
    if colors is None:
        colors = np.repeat("x", len(data_x)).astype(object)
        for i, group in enumerate(np.unique(groups)):
            colors[groups == group] = color_palette[i]
    colors = np.array(colors) if type(colors) is list else colors

    # compute correlation
    corr_data = pd.DataFrame(
        {"X": data_x,
         "Y": data_y,
         "groups": groups,
         "colors": colors}
    )
    corr_dict = {'pearson'  : pearsonr,
                 'spearman' : spearmanr,
                 'kendall'  : kendalltau}
    corr_func = corr_dict[corr_method]
    r, p = corr_func(corr_data.X, corr_data.Y)

    # plot correlation
    fig = go.Figure()

    if plot_points:
        for group in np.unique(groups):
            sub_data = corr_data[groups == group]
            sub_colors = colors[groups == group] if type(colors) == np.ndarray else colors
            textinfo = np.repeat("", sub_data.shape[0]) if textinfo is None else textinfo
            fig.add_trace(
                go.Scattergl(
                    x=sub_data.X,
                    y=sub_data.Y,
                    mode="markers",
                    marker_size=marker_size,
                    marker=dict(
                        line=dict(color="black", width=outline_width), color=sub_colors, opacity=opacity
                    ),
                    text=[
                        text + ", (" + str(np.round(x, 4)) + ", " + str(np.round(y, 4)) + ")"
                        for text, (x, y) in zip(textinfo, zip(sub_data.X, sub_data.Y))
                    ],
                    hoverinfo="text",
                    showlegend=False,
                    name=group,
                    legendgroup=group
                )
            )

    # plot lines of best fit
    if plot_fits:
        for group in np.unique(groups):
            sub_data = corr_data[groups == group]
            sub_data = sub_data.sort_values('X')
            color = corr_data[groups == group]['colors'].values[0]
            X = sub_data['X'].values
            Y = sub_data['Y'].values
            model = OLS(Y,add_constant(X)).fit()
            model_pred = model.get_prediction(add_constant(X)).summary_frame()

            fig.add_trace(
                go.Scatter(
                    x=X,
                    y=model_pred['mean'],
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    name=group,
                    legendgroup=group
                )
            )
            
            if plot_ci:
                for ci, fill in zip(['mean_ci_upper', 'mean_ci_lower'], [None, 'tonexty']):
                    fig.add_trace(
                        go.Scatter(
                            x=X,
                            y=model_pred[ci].values,
                            mode='lines',
                            line=dict(width=1),
                            marker=dict(color=color, size=0),
                            fill=fill,
                            showlegend=False,
                            name=group,
                            legendgroup=group
                        )
                    )

    # plot identity line
    if plot_identity:
        min_val = min(corr_data.X.min(), corr_data.Y.min())
        max_val = max(corr_data.X.max(), corr_data.Y.max())
        fig.add_trace(
            go.Scattergl(
                x=np.linspace(min_val, max_val),
                y=np.linspace(min_val, max_val),
                mode="lines",
                line=dict(color="black", width=line_width, dash='dash'),
                name="Identity<br>Line",
            )
        )

    # configure plot
    if same_xy_scale:
        min_val = min(corr_data.X.min(), corr_data.Y.min())
        max_val = max(corr_data.X.max(), corr_data.Y.max())
        fig.update_xaxes(
            range=(min_val - (np.abs(min_val) / 5), max_val + (max_val / 5)),
            title_text=x_title,
        )
        fig.update_yaxes(
            range=(min_val - (np.abs(min_val) / 5), max_val + (max_val / 5)),
            title_text=y_title,
        )
    else:
        fig.update_xaxes(title_text=x_title)
        fig.update_yaxes(title_text=y_title)
        if (type(x_range) is tuple) & (type(y_range) is tuple):
            fig.update_xaxes(range=x_range)
            fig.update_yaxes(range=y_range)

    fig.update_layout(
        dragmode="pan",
        font=dict(size=text_size, family=font_family),
        title_text=title
        + " r = "
        + str(np.round(r, 4))
        + ", p = "
        + str(np.round(p, 6)),
        autosize=False,
        width=plot_width,
        height=plot_height,
        showlegend=showlegend,
        template="simple_white",
    )

    # save plot
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(config=config)

# -----------------------------------------

def plotRasterAndTimeHistogram(
        raster,
        time,
        title=None,
        colorscale='gray_r',
        line_color='slategrey',
        x_title='Time (sec)',
        raster_y_title=None,
        line_y_title=None,
        plot_height=600,
        plot_width=500,
        text_size=18,
        font_family='Arial',
        dtick=None,
        save_path=None,
        plot_scale=5,
        renderer='notebook'
        ):
    '''
    Assumes an input matrix (raster) where each row is a trial and each column is a timepoint, and a time vector. Plots a
    matrix of the input matrix on top with a trial-averaged histogram below it.

    Parameters
    ==========
    raster : 2d numpy array
        numpy array representing trials in rows and time across columns, centered around 0.
    time : 1d numpy array or list
        time vector with the same length as the number of columns in raster.
    title : str
        yitle of the plot. Default is None.
    colorscale : str
        colorscale to plot the heatmap. Default is 'gray_r'.
    line_color : str
        color to plot the line. Default is 'slategrey'.
    x_title : str
        label for the x-axis. Default is 'Time (sec)'.
    raster_y_title : str
        label for the y-axis of the raster. Default is None.
    line_y_title : str
        label for the y-axis of the line. Default is None.
    plot_height, plot_width : int
        height and width of the entire plot. Defaults are 600 & 500, respectively.
    text_size : int or float
        size of all the text in the plot. Default is 18.
    font_family : str
        font family used in the plot. Default is 'Arial'.
    dtick : int or float
        delta between each tick label on the x-axis. Default is None.
    save_path : str
        file path location including filename where plot should be saved. If None, plot will not be saved. Default is None.
    plot_scale : int
        size scaling of the plot. Only used if save_path is not None and if save_path extension is of a static type. Default is 5.
    renderer : str
        plotly renderer to use for plotting. Default is 'notebook'.
    '''

    mean = raster.mean(axis=0)
    sem  = raster.std(axis=0) / np.sqrt(raster.shape[0])

    fig = make_subplots(rows=2, shared_xaxes=True, x_title=x_title, vertical_spacing=0.05)
    fig.add_trace(go.Heatmap(x=time, z=raster, colorscale=colorscale, showscale=False, showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=time, y=(mean + sem),
                             mode='lines', fill=None, line_color=line_color, hoverinfo='skip', showlegend=False, name=line_y_title, legendgroup='mean'), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=(mean - sem),
                             mode='lines', fill='tonexty', line=dict(color=line_color), hoverinfo='skip', showlegend=False, legendgroup='mean'), row=2, col=1)

    fig.update_yaxes(title_text=raster_y_title, row=1, col=1)
    fig.update_yaxes(title_text=line_y_title, row=2, col=1)
    if dtick is not None:
        fig.update_xaxes(dtick=dtick)
    # fig.update_yaxes(range=(0,np.ceil(mean)), row=2, col=1)
    fig.update_layout(template='simple_white', height=plot_height, width=plot_width, title_text=title,
                      font=dict(size=text_size, family=font_family))
    fig.update_annotations(font=dict(size=text_size+3))
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        elif save_path.split('.')[-1] != 'eps':
            fig.write_image(save_path, scale=plot_scale)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(renderer=renderer, config=config)