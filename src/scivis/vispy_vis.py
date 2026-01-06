import sys
import numpy as np

from vispy import scene, io, use
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform
from vispy.scene.cameras import Magnify1DCamera
from vispy.color import colormap, Color
from vispy import gloo
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import Qt
use('pyqt5')


class Vis:
    '''
    A general class to create a VisPy canvas and populate
    it with subplots specified at x and y grid coordinates.

    Attributes
    ==========
    title : str
        title of the overall canvas. Default is ''.
    qt_app : QApplication
        Qt application to initialize Vispy object.
    win : QWidget
        window of the Vispy app.
    layout : QVBoxLayout
        layout for Vispy app.
    width, height : int
        width and height of the entire canvas. Defaults are 1000 and 500, respectively.
    canvas : SceneCanvas
        canvas of the entire Vispy app.
    grid : Vispy grid
        Overall grid that makes up the entire Vispy canvas space.
    view_coords : dict
        dictionary where each key is the name of the given subplot and the value are the x and
        y coordinates of the subplot location in the overall grid.
    view_dict : dict
        dictionary where each key is the name of the given subplot (to match keys in view_coords) and
        the value is the associated Vispy view object of that subplot.
    grid_dict : dict
        dictionary where each key is the name of a given subplot (to match keys in view_coords) and
        the value is the associated VisPy nested grid that comprises that subplot.
    draw_ranges : dict
        dictionary where each key is the name of a given subplot (to match keys in view_coords) and
        the value is a nested dictionary of the view x/y ranges of all draw calls in that view.
    view_ranges : dict
        dictionary where each key is the name of a given subplot (to match keys in view_coords) and
        the value is a dictionary of the final x/y range to be used for that view.
    max_gl_size : int
        max texture size that your GPU supports, used to ensure all data are properly rendered.
    '''


    def __init__(
            self,
            title=None,
            width=700,
            height=700,
            bgcolor='white',
            grid_margin=0,
            grid_padding=20,
            grid_spacing=20
    ):
        '''
        Initialize Vis object

        Parameters
        =========
        title : str
            title of the overall canvas. Default is ''.
        width, height : int
            width and height of the entire canvas. Defaults are 1000 and 500, respectively.
        bgcolor : str
            color of the canvas background. Default is 'white'.
        grid_margin : int
            margin size around the outside the border of each grid. Default is 0.
        grid_padding : int
            margin size around the inside the border of each grid. Default is 20.
        grid_spacing : int
            the spacing between each grid. Default is 20.
        '''

        self.title = title
        self.qt_app = QApplication.instance() or QApplication([sys.argv])
        self.win = QWidget()
        self.win.setWindowTitle(title)
        self.layout = QVBoxLayout()
        self.win.setLayout(self.layout)

        self.width, self.height = (width, height)
        
        # VisPy canvas and view
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(self.width,self.height),
            bgcolor=bgcolor
            )
        self.canvas._send_hover_events = True
        self.layout.addWidget(self.canvas.native)
        self.grid = self.canvas.central_widget.add_grid(
            margin=grid_margin,
            padding=grid_padding,
            spacing=grid_spacing
            )

        # dictionary of all views, axes, grids, and x/y-ranges to be added
        self.view_coords = {}
        self.view_dict = {}
        self.grid_dict = {}
        self.draw_ranges = {} # all ranges in all draws for each view
        self.view_ranges = {} # final range to be used for each view

        self.max_gl_size = gloo.gl.glGetParameter(gloo.gl.GL_MAX_TEXTURE_SIZE)


    def show(self):
        '''
        Render canvas with any additional plots in a pop-up window.
        '''
        self.configure_double_click()
        print('showing')
        self.win.show()
        self.qt_app.exec_()
        print('closed')


    def save(self, save_path):
        '''
        Save the current canvas as a static image.

        Parameters
        ==========
        save_path : str
            directory with filename and extension, to which to save the image
        '''
        self.configure_double_click()
        io.imsave(save_path, self.canvas.render())
    

    def add_axes_view(
            self,
            name,
            x_label=None,
            y_label=None,
            col_span=1,
            row_span=1,
            add_xaxis=True,
            add_yaxis=True,
            magnify=False,
            mag_size_factor=10,
            mag_radius_ratio=1,
            ):
        '''
        Add x- and y-axes to a given subplot, given the view
        of that subplot.

        Parameters
        ==========
        name : str
            name of subplot to call from view_coords
        x_label, y_label : str
            titles for x- and y-axes
        col_span, y_span : int
            number of grid coordinates for this subplot to take up. Defaults are 1.
        add_xaxis, add_yaxis : bool
            whether or not to add the axis views for x- and y-axes. Defaults are True.
        magnify : bool
            whether or not to use the Magnify1DCamera view camera for this subplot. Default is False.
        mag_size_factor : int
            by what factor to magnify. Only used if magnify=True. Default is 10.
        mag_radius_ratio : int
            radius of the magnification. Only used if magnify=True. Default is 1.
        '''

        row, col = self.view_coords[name]
        nested_grid = self.grid.add_grid(row, col, row_span=row_span, col_span=col_span)
        nested_grid.spacing = 0

        # Add X and Y axes
        axis_kwargs = dict(
            text_color='black',
            tick_color='black',
            tick_width=1,
            axis_width=2,
            major_tick_length=5,
            minor_tick_length=3
        )

        if add_xaxis:
            xaxis = scene.AxisWidget(orientation='bottom', axis_label=x_label, **axis_kwargs)
            xaxis.height_max = 40
            nested_grid.add_widget(xaxis, row=1, col=1 if add_yaxis else 0)
        
        if add_yaxis:
            yaxis = scene.AxisWidget(orientation='left', axis_label=y_label, **axis_kwargs)
            yaxis.width_max = 40
            nested_grid.add_widget(yaxis, row=0, col=0)

        # Link axes to the view
        view = nested_grid.add_view(row=0, col=1 if add_yaxis else 0)
        view.border_color = 'black'
        if magnify:
            view.camera = Magnify1DCamera(mag=1, size_factor=mag_size_factor, radius_ratio=mag_radius_ratio)
        else:
            view.camera = scene.cameras.PanZoomCamera()

        if add_xaxis:
            xaxis.link_view(view)
        if add_yaxis:
            yaxis.link_view(view)
        
        self.grid_dict[name] = nested_grid

        return view


    def scale_view(
            self,
            name
    ):
        '''
        Given the name of a view, compute the min and max ranges
        of this view based on all the data currently plotted in the
        view, and update the range of the view.
        '''
        # compute final view_range for this view
        self.view_ranges[name] = dict(
            x=(np.min([val['x'][0] for val in self.draw_ranges[name].values()]),
               np.max([val['x'][1] for val in self.draw_ranges[name].values()])),
            y=(np.min([val['y'][0] for val in self.draw_ranges[name].values()]),
               np.max([val['y'][1] for val in self.draw_ranges[name].values()])))

        # update view range
        if any([isinstance(ch, scene.Image) for ch in self.view_dict[name].scene.children]):
            self.view_dict[name].camera.rect = (
                self.view_ranges[name]['x'][0],
                self.view_ranges[name]['y'][0],
                self.view_ranges[name]['x'][1],
                self.view_ranges[name]['y'][1]
                )
        else:
            self.view_dict[name].camera.set_range(
                x=self.view_ranges[name]['x'],
                y=self.view_ranges[name]['y']
                )


    def configure_double_click(
            self
    ):
        '''
        Compute and assign the value ranges to reset for each
        view when a double click is detected.
        '''
        def on_mouse_double_click(event):
            for name in self.view_dict.keys():
                self.scale_view(name=name)

        # attach to double click and initialize scaling
        self.canvas.events.mouse_double_click.connect(on_mouse_double_click)
        on_mouse_double_click(0)


    def on_key_press(
            self,
            event
            ):
        '''
        If canvas has a slider, allow slider to be controlled by left and right arrow keys.

        Parameters
        ==========
        event : event object
            which keyboard event just occurred. Only ones of consequence are "Left" and "Right".
        '''
        cur_slider_val = self.slider.value()
        if event.key == 'Left':
            self.slider.setValue(max(self.slider_range[0], cur_slider_val - 1))
        if event.key == 'Right':
            self.slider.setValue(min(cur_slider_val + 1, self.slider_range[1]))
        else:
            pass
    

    def add_slider(
            self,
            slider_size
    ):
        '''
        Add a slider widget to the Vis object.

        Parameters
        ==========
        slider_size : int
            number of slider positions to create in the slider
        '''
        self.slider = QSlider(Qt.Horizontal)
        self.slider_range = (0, slider_size-1)
        self.slider.setRange(self.slider_range[0],self.slider_range[1])
        self.slider.setValue(0)
        self.layout.addWidget(self.slider)


    def draw_colorbar(
            self,
            name,
            cmap,
            clim,
            orientation='right',
            border_width=1
    ):
        '''
        ## NOT CURRENTLY IMPLEMENTED ##
        # because the default spacing in the grid is strangely
        # far from the image
        '''
        cbar_widget = scene.ColorBarWidget(
            clim=clim,
            cmap=cmap,
            orientation=orientation,
            border_width=border_width,
            pos=(0,0)
        )
        cbar_widget.border_color = 'black'

        self.grid_dict[name].add_widget(cbar_widget)


    def draw_multiscale_image(
            self,
            data,
            view,
            cmap='viridis'
    ):
        '''
        If a matrix is larger than the GL_MAX_TEXTURE_SIZE limit, use this
        to plot the same image as multiple smaller images tiled together.

        Parameters
        ==========
        data : 2d array or list of 2d arrays
            if 2d array, image will be plotted. If a list of 2d arrays, a slider will
            be created and each 2d array in the list will be plotted along the slider.
        view : VisPy view object
            a view associated with a nested grid, into which this visualization will
            be placed.
        cmap : str
            which colormap to use to color the image.
        '''

        x_len = data.shape[1]
        y_len = data.shape[0]
        clim = (data.min(), data.max())

        # compute tile indices
        x_tiles = np.ceil(x_len / self.max_gl_size)
        y_tiles = np.ceil(y_len / self.max_gl_size)
        x_indices = np.array_split(np.arange(data.shape[1]), x_tiles)
        y_indices = np.array_split(np.arange(data.shape[0]), y_tiles)

        # plot all tiles
        image_ls = []
        for x_tile in x_indices:
            for y_tile in y_indices:
                image = scene.Image(
                    data[np.ix_(y_tile,x_tile)].astype(np.float32),
                    clim=clim,
                    cmap=cmap,
                    parent=view.scene
                    )
                image.transform = STTransform(translate=(x_tile[0],y_tile[0]))
                image_ls.append(image)

    
    def draw_image(
            self,
            data,
            name='image',
            title_ls=None,
            scale_image=True,
            cmap='viridis',
            global_scaling=False,
            **kwargs
    ):
        '''
        Plot image data, given a 2d image or a list of 2d images.

        Parameters
        ==========
        data : 2d array or list of 2d arrays
            if 2d array, image will be plotted. If a list of 2d arrays, a slider will
            be created and each 2d array in the list will be plotted along the slider.
        name : str
            name of the view, if previously added to view_coords. If view_coords is
            empty, a single sub-grid at (0,0) will be created.
        title_ls : list of str
            if data is a list of 2d arrays, provide a list of titles for each step of
            the slider. If None, titles will become 'Step X' for each step. Default is None.
        scale_image : bool
            whether or not to allow the image to stretch as canvas size changes. If False,
            the aspect ratio of the image will be made 1.
        cmap : str
            which colormap to use to color the image.
        **kwargs
            extra keyword arguments to be fed into self.add_axes_view()
        '''

        # if no view has previously been set, create a single sub-grid for this plot
        if len(self.view_coords) == 0:
            self.view_coords[name] = (0, 0)

        if name not in self.view_dict.keys():
            view = self.add_axes_view(name=name, **kwargs)
            self.view_dict[name] = view
            self.draw_ranges[name] = {}
        
        draw_num = len(self.view_dict[name].scene.children) - 1

        # set current data
        if type(data) is not list:
            cur_data = data.astype(np.float32)
        else:
            cur_data = data[0].astype(np.float32)

        # plot image
        if cur_data.size > self.max_gl_size:
            print('data size > max_gl_size; plotting multiscale')
            self.draw_multiscale_image(data=cur_data, view=self.view_dict[name], cmap=cmap)
        else:
            image = scene.Image(data=cur_data, parent=self.view_dict[name].scene, cmap=cmap)

        # if data is a list, create a slider to pan through it
        if type(data) is list:
            if title_ls is None:
                title_ls = [f'Step {x}' for x in range(len(data))]
            self.add_slider(slider_size=len(data))

            if global_scaling:
                x_max = np.max([im.shape[1] for im in data])
                y_max = np.max([im.shape[0] for im in data])
                self.draw_ranges[name][draw_num] = dict(x=(0, x_max),
                                                        y=(0, y_max))

            def update_plot(index):
                cur_data = data[index].astype(np.float32)
                if cur_data.size > self.max_gl_size:
                    for visual in list(self.view_dict[name].scene.children):
                        if type(visual) is scene.Image:
                            visual.parent = None
                    self.draw_multiscale_image(data=cur_data, view=self.view_dict[name], cmap=cmap)
                else:
                    image.set_data(cur_data)
                
                if not global_scaling:
                    self.draw_ranges[name][draw_num] = dict(x=(0, cur_data.shape[1]),
                                                            y=(0, cur_data.shape[0]))
                self.win.setWindowTitle(title_ls[index])
                self.scale_view(name=name)
                self.canvas.update()
            self.slider.valueChanged.connect(update_plot)
            self.canvas.events.key_press.connect(self.on_key_press)
            update_plot(0)
        else:
            self.draw_ranges[name][draw_num] = dict(x=(0, cur_data.shape[1]),
                                                    y=(0, cur_data.shape[0]))

        if not scale_image:
            self.view_dict[name].camera.aspect=1


    def draw_line(
            self,
            y,
            x=None,
            name='line',
            title_ls=None,
            line_color='black',
            global_scaling=False,
            **kwargs
    ):
        '''
        Plot line data, given 1d arrays for x and y values, or a list
        of 1d x and y arrays.

        Parameters
        ==========
        y : 1d array or list of 1d arrays
            if 1d array, will plot single line. If list of 1d arrays, a slider will
            be created and each 1d array in the list will be plotted along the slider.
        x : 1d array or list or 1d arrays
            associated x-axis values for y. If provided, len(x) must match len(y). If
            None, x will be inferred as np.arange(0, len(y)) for each provided y.
        name : str
            name of the view, if previously added to view_coords. If view_coords is
            empty, a single sub-grid at (0,0) will be created.
        title_ls : list of str
            if data is a list of 1d arrays, provide a list of titles for each step of
            the slider. If None, titles will become 'Step X' for each step. Default is None.
        line_color : str or list of str
            color of the line, or a list of colors if y is a list. Default is 'black'.
        global_scaling : bool
            if False, re-scale y-axis when panning through slider. Only used if y is
            a list. Default is False.
        **kwargs
            extra keyword arguments to be fed into self.add_axes_view()
        '''

        # if no view has previously been set, create a single sub-grid for this plot
        if len(self.view_coords) == 0:
            self.view_coords[name] = (0, 0)

        if name not in self.view_dict.keys():
            view = self.add_axes_view(name=name, **kwargs)
            self.view_dict[name] = view
            self.draw_ranges[name] = {}
        
        draw_num = len(self.view_dict[name].scene.children) - 1
        
        # if x is not provided, create x as a range from 0 to N for each y
        if x is None:
            if type(y) is list:
                x = [np.arange(0,len(cur_y)) for cur_y in y]
            else:
                x = np.arange(0,len(y))
        
        # set current data
        if type(y) is not list:
            cur_y = y.astype(np.float32)
            cur_x = x.astype(np.float32)
        else:
            cur_y = y[0].astype(np.float32)
            cur_x = x[0].astype(np.float32)
        
        # set line color
        if type(line_color) is list:
            cur_line_color = line_color[0]
        else:
            cur_line_color = line_color

        # plot line
        line = scene.Line(pos=np.column_stack((cur_x, cur_y)),
                          color=cur_line_color, width=1, 
                          parent=self.view_dict[name].scene)
        
        # if data is a list, create a slider to pan through it
        if type(y) is list:
            if title_ls is None:
                title_ls = [f'Step {x}' for x in range(len(y))]
            self.add_slider(slider_size=len(y))

            if global_scaling: # same scale across all slider locations
                x_min = np.min([cur_x.min() for cur_x in x])
                x_max = np.max([cur_x.max() for cur_x in x])
                y_min = np.min([cur_y.min() for cur_y in y])
                y_max = np.max([cur_y.max() for cur_y in y])
                self.draw_ranges[name][draw_num] = dict(x=(x_min, x_max),
                                                        y=(y_min, y_max))

            def update_plot(index):
                nonlocal cur_x, cur_y
                cur_y = y[index].astype(np.float32)
                cur_x = x[index].astype(np.float32)

                nonlocal cur_line_color
                if type(line_color) is list:
                    cur_line_color = line_color[index]

                line.set_data(np.column_stack((cur_x, cur_y)),
                                color=cur_line_color)
                
                self.win.setWindowTitle(title_ls[index])
                self.canvas.update()

                # update view range
                if not global_scaling:
                    self.draw_ranges[name][draw_num] = dict(x=(cur_x.min(), cur_x.max()),
                                                            y=(cur_y.min(), cur_y.max()))
                self.scale_view(name=name)
                

            self.slider.valueChanged.connect(update_plot)
            self.canvas.events.key_press.connect(self.on_key_press)
            update_plot(0)
        else:
            self.draw_ranges[name][draw_num] = dict(x=(cur_x.min(), cur_x.max()),
                                                    y=(cur_y.min(), cur_y.max()))



    def draw_scatter(
            self,
            y,
            x=None,
            name='scatter',
            title_ls=None,
            marker_color='white',
            opacity=1,
            marker_size=5,
            outline_width=1,
            outline_color='black',
            marker_symbol='o',
            global_scaling=False,
            **kwargs
    ):
        '''
        Plot a scatter plot given a pair of x and y values, or a list of scatter plots
        given a list of x and y pairs.

        Parameters
        ==========
        y : 1d array or list of 1d arrays
            if 1d array, will plot single line. If list of 1d arrays, a slider will
            be created and each 1d array in the list will be plotted along the slider.
        x : 1d array or list or 1d arrays
            associated x-axis values for y. If provided, len(x) must match len(y). If
            None, x will be inferred as np.arange(0, len(y)) for each provided y.
        name : str
            name of the view, if previously added to view_coords. If view_coords is
            empty, a single sub-grid at (0,0) will be created.
        title_ls : list of str
            if data is a list of 1d arrays, provide a list of titles for each step of
            the slider. If None, titles will become 'Step X' for each step. Default is None.
        marker_color : str
            color of the datapoints, represented as a str (word or hex) or a tuple of RGB values. If 
            a list/array of values, it will be interpreted that each color represents each x/y value in order.
        opacity : float
            a floating number from [0,1] designating the transparency of the scatterpoints. Default is 1.
        marker_size : int or float
            size of the datapoints. Default is 5.
        outline_width : int or float
            width of the line outlining each scatterpoint. Default is 1.
        outline_color : str
            color of the line outlining each scatterpoint. Default is 'black'.
        marker_symbol : str
            the styling of each scatterpoint. Default is 'o'.
        global_scaling : bool
            if False, re-scale y-axis when panning through slider. Only used if y is
            a list. Default is False.
        **kwargs
            extra keyword arguments to be fed into self.add_axes_view()
        '''
        
        # if no view has previously been set, create a single sub-grid for this plot
        if len(self.view_coords) == 0:
            self.view_coords[name] = (0, 0)

        if name not in self.view_dict.keys():
            view = self.add_axes_view(name=name, **kwargs)
            self.view_dict[name] = view
            self.draw_ranges[name] = {}
        
        draw_num = len(self.view_dict[name].scene.children) - 1

        # if x is not provided, create x as a range from 0 to N for each y
        if x is None:
            if type(y) is list:
                x = [np.arange(0,len(cur_y)) for cur_y in y]
            else:
                x = np.arange(0,len(y))

        # set current data
        if type(y) is not list:
            cur_y = y.astype(np.float32)
            cur_x = x.astype(np.float32)
        else:
            cur_y = y[0].astype(np.float32)
            cur_x = x[0].astype(np.float32)
        
        # configure colors
        if type(marker_color) is list:
            assert len(marker_color) == len(cur_y), \
                "If marker_color is a list, len(marker_color) must equal len(cur_y)."
            marker_color = [Color(c, alpha=opacity) for c in marker_color]
        else:
            marker_color = Color(marker_color, alpha=opacity)

        # plot scatter
        scatter = visuals.Markers(parent=self.view_dict[name].scene)
        scatter.set_data(
            pos=np.array([cur_x,cur_y]).T,
            edge_width=outline_width,
            edge_color=outline_color,
            face_color=marker_color,
            size=marker_size,
            symbol=marker_symbol
            )
        scatter.antialias = 0
        scatter.alpha = opacity
        scatter.set_gl_state('translucent')

        # if data is a list, create a slider to pan through it
        if type(y) is list:
            if title_ls is None:
                title_ls = [f'Step {x}' for x in range(len(y))]
            self.add_slider(slider_size=len(y))

            if global_scaling: # same scale across all slider locations
                x_min = np.min([cur_x.min() for cur_x in x])
                x_max = np.max([cur_x.max() for cur_x in x])
                y_min = np.min([cur_y.min() for cur_y in y])
                y_max = np.max([cur_y.max() for cur_y in y])
                self.draw_ranges[name][draw_num] = dict(x=(x_min, x_max),
                                                        y=(y_min, y_max))

            def update_plot(index):
                nonlocal cur_x, cur_y
                cur_x = x[index].astype(np.float32)
                cur_y = y[index].astype(np.float32)

                scatter.set_data(pos=np.array([cur_x,cur_y]).T,
                                 edge_width=outline_width,
                                 edge_color=outline_color,
                                 face_color=marker_color,
                                 size=marker_size,
                                 symbol=marker_symbol
                                 )
                
                self.win.setWindowTitle(title_ls[index])
                self.canvas.update()

                # update view range
                if not global_scaling:
                    self.draw_ranges[name][draw_num] = dict(x=(cur_x.min(), cur_x.max()),
                                                            y=(cur_y.min(), cur_y.max()))
                self.scale_view(name=name)
            self.slider.valueChanged.connect(update_plot)
            self.canvas.events.key_press.connect(self.on_key_press)
            update_plot(0)
        else:
            self.draw_ranges[name][draw_num] = dict(x=(cur_x.min(), cur_x.max()),
                                                    y=(cur_y.min(), cur_y.max()))


    def draw_3d(
            self,
            x,
            y,
            z,
            name='3d',
            plot_mode='scatter',
            show_3d_axes=True,
            marker_color='white',
            marker_size=5,
            outline_width=1,
            outline_color='black',
            marker_symbol='o',
            opacity=1,
            line_color='black',
            line_width=2,
            **kwargs
    ):
        '''
        Plot a 3d either line or scatter plot, given equal length x, y, and z vectors.
        Note: This does not support plotting with a slider.

        Parameters
        ==========
        x, y, z : 1d arrays
            equivalent length 1d arrays to be plotted in 3d.
        name : str
            name of the view, if previously added to view_coords. If view_coords is
            empty, a single sub-grid at (0,0) will be created.
        plot_mode : str
            one of 'scatter' or 'line' to plot either a scatter or line plot. Default is 'scatter'.
        show_3d_axes : bool
            whether or not to show 3d axes in the 3d plot. Default is True.
        marker_color : str
            color of the datapoints, represented as a str (word or hex) or a tuple of RGB values. If 
            a list/array of values, it will be interpreted that each color represents each x/y/z value
            in order. Only used if plot_mode=='scatter'. Default is 'white'.
        marker_size : int/float or list/1d array
            size of the datapoints. If list/array, must be the same length as x/y/z. Only used if
            plot_mode=='scatter'. Default is 5.
        outline_width : int or float
            width of the line outlining each scatterpoint. Only used if plot_mode=='scatter'. Default is 1.
        outline_color : str
            color of the line outlining each scatterpoint. ONly used if plot_mode=='scatter'. Default is 'black'.
        marker_symbol : str
            symbol to be used for the scatterpoints. Only used if plot_mode=='scatter'. Default is 'o'.
        opacity : float
            a floating number from [0,1] designating the transparency of the scatterpoints. Only used
            if plot_mode=='scatter'. Default is 1.
        line_color : str or list
            color of the line. If list, must be length of x/y/z. Only used if plot_mode=='line'. Default is 'black'.
        line_width : int or float
            width of the line. Only used if plot_mode=='line'. Default is 2.
        '''
        
        # if no view has previously been set, create a single sub-grid for this plot
        if len(self.view_coords) == 0:
            self.view_coords[name] = (0, 0)

        if name not in self.view_dict.keys():
            view = self.add_axes_view(name=name, **kwargs)
            self.view_dict[name] = view
            self.draw_ranges[name] = {}
        
        draw_num = len(self.view_dict[name].scene.children) - 1
        
        # plot line or scatter
        if plot_mode.lower() == 'line':
            data = scene.Line(pos=np.column_stack((x,y,z)),
                              color=line_color, width=line_width,
                              parent=self.view_dict[name].scene)
        elif plot_mode.lower() == 'scatter':
            data = visuals.Markers(parent=self.view_dict[name].scene)
            data.set_data(
                pos=np.array([x,y,z]).T,
                edge_width=outline_width,
                edge_color=outline_color,
                face_color=marker_color,
                size=marker_size,
                symbol=marker_symbol
                )
            data.antialias = 0
            data.alpha = opacity
            data.set_gl_state('translucent')
        else:
            raise Exception("Invalid plot_mode. Must be one of ['line', 'scatter'].")

        self.view_dict[name].add(data)
        self.view_dict[name].camera = 'turntable'

        if show_3d_axes:
            axis = visuals.XYZAxis(
                parent=self.view_dict[name].scene,
                width=2,
                # color='black',
                )

        self.canvas.update()
        self.draw_ranges[name][draw_num] = dict(x=(x.min(), x.max()),
                                                y=(y.min(), y.max()))
