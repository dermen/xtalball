#!/usr/bin/python
try: 
    import Tkinter as tk
except ImportError:
    import tkinter as tk
import sys
import re
import os
import pandas
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

import pylab as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from range_slider import RangeSlider

btnstyle = {'font': 'Helvetica 10 bold', 
            'activebackground': 'green', 'activeforeground': 'white',
            'relief': tk.RAISED, 'highlightcolor':'red'}  # , 'bd':5}
labstyle = {'font': 'Helvetica 14 bold', 'bg': 'snow', 'fg': 'black',
            'activebackground': 'green', 'activeforeground': 'white'}

fr = {'bg': None}
frpk = {'padx': 5, 'pady': 5}

class Formatter(object):
    """
    Mouse hover and see inensity in bottom right of frame 
    From:
    https://stackoverflow.com/a/27707723/2077270
    """
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.0f}, y={:.0f}, z={:.01f}'.format(x, y, z)

class ZoomPan:
    """
    scroll wheel zoom functionality (dont really use this)
    Borrowed from
    https://stackoverflow.com/a/19829987/2077270
    """
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None

    def zoom_factory(self, ax, base_scale = 2.):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + \
                new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + \
                new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event',onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion

class ImageViewer(tk.Frame):
    """
    Main image viewer ; uses range_slider.py to update clim
    """
    def __init__(self, master, img_data,  
        passive=False, *args, **kwargs):
        
        tk.Frame.__init__(self, master,  background='white') #*args, **kwargs)
        self.master = master
        
        self.image_frame = tk.Frame( self.master, **fr )
        self.image_frame.pack( side=tk.TOP)
        
        self.slider_frame = tk.Frame(self.master, **fr)
        self.slider_frame.pack(side=tk.TOP, expand=1, fill=tk.BOTH)
        self.hist_frame = tk.Frame( self.slider_frame , **fr)
        self.hist_frame.pack( side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)
        self.vmin = self.vmax = None

        #load the image
        self.img = img_data
        
        self._create_figure()    
        self._add_img()
        self._setup_canvas()
        self._zoom_im = None

        self._add_hist_updater()
        self._update_clim()

    def set_data(self,data):
        self.ax.images[0].set_data(data)

    def _update_clim(self):
        self.vmin, self.vmax = self.hist_updater.minval, self.hist_updater.maxval
        self._im.set_clim( vmin=self.vmin, vmax=self.vmax)
        self.canvas.draw()
        self.master.after( 500, self._update_clim )    

    def _add_hist_updater(self):
        self.hist_updater = RangeSlider( self.hist_frame, range_slider_len=800, 
            minval=-100, maxval=1000, background='white')
        self.hist_updater.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        
    def _create_figure(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        
        #self.ax.set_aspect('equal')
        self.fig.patch.set_visible(False)
        self.ax.axis('off')
    
    def _add_img(self):
        self._im = self.ax.imshow(self.img, interpolation='nearest', 
            norm=None, 
            vmin=self.vmin, 
            vmax=self.vmax, 
            cmap='gist_gray')
        self.vmin,self.vmax = self._im.get_clim()
        #self.cbar = plt.colorbar( self._im)
        self.ax.format_coord = Formatter(self._im)

    def _setup_canvas(self):
        toplvl= tk.Toplevel(self.master)
        self.disp_frame = tk.Frame(toplvl)
        self.disp_frame.pack(side=tk.TOP, expand=1, fill=tk.BOTH, **frpk)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.disp_frame) 
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, 
            expand=1, **frpk)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, 
            self.disp_frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, 
            fill=tk.BOTH, expand=1, **frpk)

        self.canvas.draw()
        
        self.zp = ZoomPan()
        self.figZoom = self.zp.zoom_factory( self.ax, base_scale=1.1)









