import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def animated_basic(x, uinit, func, Nt, *param, plot_dim = None):
    """
    Return a matplotlib 2D animation of the PDE evolution.

    Parameters
    ----------
    x : 1d numpy array
        Spatial grid of the system.
    uinit : 1d numpy array.
        Initial condition of the PDE (a function evaluated on x).
    func : function
        Function that evaluate the next temporal step of the solution u.
        Here you can pass the implementation of the LAX methods.
        The parameters of the function needs to be: func(u, *param) where
        u is the solution at time i and *param are the other parameters of the 
        system (like alpha and ninner for the implemented LAX).
    Nt : int
        Number of temporal step for the evolution of the PDE.
    *param : *params argument
        This values will be passed to func, for example you can put here
        alpha, ninner in the case of the LAX method.
    plot_dim : int or None
        The dimension to plot in case of multidimensional problem. If None
        the problem is assumed 1D.
    
    Returns
    -------
    """
    def animate(i, plot_dim):
        # LAX goes directly here
        if plot_dim == None:
            line.set_ydata( func(u, *param) )  # update the data
        else:
            line.set_ydata( func(u, *param)[plot_dim] )  # update the data
        return line,
    
    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    u = np.copy(uinit)
    u1 = np.copy(uinit)
    fig, ax1 = plt.subplots()
    if plot_dim == None:
        line, = ax1.plot(x, uinit)
    else:
        line, = ax1.plot(x, uinit[plot_dim])

    ani = animation.FuncAnimation(fig, animate, np.arange(0, Nt), init_func=init, fargs = (plot_dim,),
                                  interval=25, blit=True)
    plt.show()


def animated_full(func, func_mesh, x, t, uinit, *param, imported_title = None, plot_dim = None):
    """
    Return a2D animation of the PDE evolution and a 3D surface of the evoluton
    int time with teh evolving line moving on it.

    Parameters
    ----------
    func : function
        Function that evaluate the next temporal step of the solution u.
        Here you can pass the implementation of the LAX methods.
        The parameters of the function needs to be: func(u, *param) where
        u is the solution at time i and *param are the other parameters of the 
        system (like alpha and ninner for the implemented LAX).
    func_mesh : function
        Function that evaluate all the temporal step of the solution u. The
        steps need to be putted in a grid by the function in orther to be 
        plotted in a surface plot.
        Here you can pass the implementation of the LAX methods.
        The parameters of the function needs to be: func(mesh, uinit, Nt, *param) 
        where mesh is the grid (numpy 2d matrix of size((len(x), len(t))), 
        uinit is the solution at initial time, Nt is Number of temporal step for
        the evolution of the PDE,*param are the other parameters of the system 
        (like alpha and ninner for the implemented LAX).
    x : 1d numpy array
        Spatial grid of the system.
    t : 1d numpy array
        temporal grid of the system.
    uinit : 1d numpy array.
        Initial condition of the PDE (a function evaluated on x).
        Nt : int
        Number of temporal step for the evolution of the PDE.
    *param : *params argument
        This values will be passed to func, for example you can put here
        alpha, ninner in the case of the LAX method.
    imported_title : string
        The title of the figure, if None the function will set this automatic.

    Returns
    -------
    """

    if imported_title == None:
        imported_title = 'Animated plot'
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    Nt = len(t)
    n = len(x)

    duration = Nt

    u = np.copy(uinit)
    X, Y = np.meshgrid(x, t)
    evo = np.empty((Nt, n))
    func_mesh(evo, np.copy(uinit), Nt, *param) 

    ## Define update functions #######
    def my_surface(evo, X, Y, i, n, dx):
        startt = i * dt
        endt = (i+1) * dt
        contoursy = {"show" : True, "start": startt, "end": endt, "size": 2*dt, "width" : 3}
        return go.Surface( z=evo, x=X, y=Y, opacity = 0.4, contours = {'y': contoursy})

    def my_scatter(x, func, u, plot_dim, *param):
        if plot_dim == None: return go.Scatter(x = x, y = func(u, *param), mode = "lines")
        else: return go.Scatter(x = x, y = func(u, *param)[plot_dim], mode = "lines")

    ######### Definire un subplot con Plolty
    fig = make_subplots( rows=1, cols=2, 
                         subplot_titles=('Title1', 'Title2'),
                         horizontal_spacing=0.051,
                         specs=[[{"type": "scatter"} , {"type": "surface"}]]
    )

    ## subplot 1 #######
    if plot_dim == None: init_scatter = go.Scatter( x=x, y=uinit, mode='lines')
    else: init_scatter = go.Scatter( x=x, y=uinit[plot_dim], mode='lines')
    fig.add_trace(init_scatter, row = 1, col = 1)
    fig.update_xaxes(title_text = "x")
    fig.update_yaxes(title_text = "u")
    m = np.min(uinit)
    M = np.max(uinit)
    fig.update_layout(yaxis_range=[m , M ])
 

    ## subplot 2 #######
    fig.add_trace(my_surface(evo, X, Y, 0, n, dx), row = 1, col = 2)

    ## Evolving the frames ###############################################
    my_frames = [go.Frame(data = [my_scatter(x, func, u, plot_dim, *param),
                                  my_surface(evo, X, Y, i, n, dx)], 
                traces=[0,1], name=str(i), layout_yaxis_range=[m , M])
                for i in range(Nt)]

    fig.frames = my_frames

    ## Setting dello slider ##
    def frame_args(duration):
        return dict(frame={"duration": duration},
                    mode ="immediate",
                    fromcurrent=True,
                    transition={"duration": duration, 
                                "easing": "linear"},)
    # Definire i parametri dello slider 
    sliders = [{"pad": {"b": 10, "t": 60},
                "len": 0.5,
                "x": 0.1,
                "y": 0,
                "steps": [{"args": [[f.name], frame_args(duration)],
                           "label": str(k),
                           "method": "animate",}
                    for k, f in enumerate(fig.frames)],}]

    ## Layout generale della figura ##
    fig.update_layout(
         title=imported_title,
         width=1200,
         height=700,
         scene=dict(
             xaxis_title='x',
             yaxis_title='t',
             zaxis_title='u',
             aspectratio=dict(x=1, y=1, z=1)),
         updatemenus = [
                {
                "buttons": [
                    {"args": [None, frame_args(duration)],
                     "label": "&#9654;", # play symbol
                     "method": "animate",},
                    {"args": [[None], frame_args(duration)],
                     "label": "&#9724;", # pause symbol
                     "method": "animate",
                    },],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
         ],
         sliders=sliders
    )

    fig.show()
