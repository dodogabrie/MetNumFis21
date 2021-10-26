import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def animated_basic(func, uinit, x, Nt, param, plot_dim = None, dilat_size = 0.2):
    """
    Return a matplotlib 2D animation of the PDE evolution.

    Parameters
    ----------
    func : function
        Function that evaluate the next temporal step of the solution u.
        Here you can pass the implementation of the LAX methods.
        The parameters of the function needs to be: func(u, *param) where
        u is the solution at time i and *param are the other parameters of the
        system (like alpha and ninner for the implemented LAX).
    uinit : 1d numpy array.
        Initial condition of the PDE (a function evaluated on x).
    x : 1d numpy array
        Spatial grid of the system.
    Nt : int
        Number of temporal step for the evolution of the PDE.
    param : tuple
        Values will be passed to func in a tuple (excluded u), for example you
        can put here (alpha, ninner) in the case of the LAX method.
    plot_dim : int or None
        The dimension to plot in case of multidimensional problem. If None
        the problem is assumed 1D.
    dilat_size : float
        The dilatation of the ylim respect to the max and the min of the
        function at initial time.

    Returns
    -------
    """
    def animate(i, plot_dim):
        # LAX goes directly here
        if plot_dim == None: line.set_ydata( func(u, *param) )  # update the data
        else: line.set_ydata( func(u, *param)[plot_dim] )  # update the data
        return line,

    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    u = np.copy(uinit)
    u1 = np.copy(uinit)
    fig, ax1 = plt.subplots()
    if plot_dim == None:
        min_u = np.min(uinit)
        max_u = np.max(uinit)
        line, = ax1.plot(x, uinit)
    else:
        min_u = np.min(uinit[plot_dim])
        max_u = np.max(uinit[plot_dim])
        line, = ax1.plot(x, uinit[plot_dim])

    size = (max_u - min_u)*dilat_size
    ax1.set_ylim(min_u - size, max_u + size)

    ani = animation.FuncAnimation(fig, animate, np.arange(0, Nt), init_func=init, fargs = (plot_dim,),
                                  interval=25, blit=True)
    plt.show()

def animated_with_slider(func, uinit, x, Nt, dt, param, plot_dim = None, dilat_size = 0.1, title = None):
    """
    Return a matplotlib 2D animation of the PDE evolution.

    Parameters
    ----------
    func : function
        Function that evaluate the next temporal step of the solution u.
        Here you can pass the implementation of the LAX methods.
        The parameters of the function needs to be: func(u, *param) where
        u is the solution at time i and *param are the other parameters of the
        system (like alpha and ninner for the implemented LAX).
    uinit : 1d numpy array.
        Initial condition of the PDE (a function evaluated on x).
    x : 1d numpy array
        Spatial grid of the system.
    Nt : int
        Number of temporal step for the evolution of the PDE.
    dt : float
        Temporal step of the simulation.
    param : tuple
        Values will be passed to func in a tuple (excluded u), for example you
        can put here (alpha, ninner) in the case of the LAX method.
    plot_dim : int or None
        The dimension to plot in case of multidimensional problem. If None
        the problem is assumed 1D.
    dilat_size : float
        The dilatation of the ylim respect to the max and the min of the
        function at initial time.
    title: string
        Title of the figure.

    Returns
    -------
    """

    if title == None: title = 'Animated plot'

    duration = Nt

    u = np.copy(uinit)
    utest = np.copy(uinit)

    def my_scatter(x, func, u, plot_dim, *param):
        if plot_dim == None: return go.Scatter(x = x, y = func(u, *param), mode = "lines+markers", marker=dict(size=6))
        else: return go.Scatter(x = x, y = func(u, *param)[plot_dim], mode = "lines+markers", marker=dict(size=6))

    ######### Definire una figure con Plolty
    if plot_dim == None: init_scatter = go.Scatter( x=x, y=uinit, mode='lines+markers', marker=dict(size=6))
    else: init_scatter = go.Scatter( x=x, y=uinit[plot_dim], mode='lines+markers', marker=dict(size=6))

    for i in range(Nt):
        func(utest, *param)

    fig = go.Figure(data = init_scatter)
    fig.update_xaxes(title_text = "x")
    fig.update_yaxes(title_text = "u")
    if plot_dim == None:
        m = min(np.min(utest), np.min(uinit))
        M = max(np.max(utest), np.max(uinit))
    else:
        m = min(np.min(utest[plot_dim]),np.min(uinit[plot_dim]))
        M = max(np.max(utest[plot_dim]), np.max(uinit[plot_dim]))
    dilat = (M - m) * dilat_size
    fig.update_layout(yaxis_range=[m - dilat , M + dilat ])


    ## Evolving the frames ###############################################
    my_frames = [go.Frame(data = [my_scatter(x, func, u, plot_dim, *param)],
                name=str(i), layout_yaxis_range=[m - dilat , M + dilat])
                for i in range(Nt)]

    fig.frames = my_frames

    _fig_update_layout(fig, duration, title)

    fig.show()
    return


def animated_full(func, x, t, uinit, param, title = None, plot_dim = None, dilat_size = 0.1):
    """
    Return a 2D animation of the PDE evolution and a 3D surface of the evoluton
    int time with teh evolving line moving on it.

    Parameters
    ----------
    func : function
        Function that evaluate the next temporal step of the solution u.
        Here you can pass the implementation of the LAX methods.
        The parameters of the function needs to be: func(u, *param) where
        u is the solution at time i and param are the other parameters of the
        system (like alpha and ninner for the implemented LAX).
    x : 1d numpy array
        Spatial grid of the system.
    t : 1d numpy array
        temporal grid of the system.
    uinit : 1d numpy array.
        Initial condition of the PDE (a function evaluated on x).
        Nt : int
        Number of temporal step for the evolution of the PDE.
    param : tuple
        Values passed to func (excluded u), for example you can put here
        (alpha, ninner) in the case of the LAX method.
    title : string
        The title of the figure, if None the function will set this automatic.

    Returns
    -------
    """

    if title == None:
        title = 'Animated plot'
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    Nt = len(t)
    n = len(x)

    duration = Nt

    u = np.copy(uinit)
    X, Y = np.meshgrid(x, t)
    evo = np.empty((Nt, n))
    evo[0] = u[plot_dim]
    for i in range(Nt-1):
        u = func(u, *param)
        evo[i+1] = u[plot_dim]
    u = np.copy(uinit)

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
                         subplot_titles=('Evolution in time', 'Surface with evolution'),
                         horizontal_spacing=0.051,
                         specs=[[{"type": "scatter"} , {"type": "surface"}]]
    )

    ## subplot 1 #######
    if plot_dim == None: init_scatter = go.Scatter( x=x, y=uinit, mode='lines')
    else: init_scatter = go.Scatter( x=x, y=uinit[plot_dim], mode='lines')
    fig.add_trace(init_scatter, row = 1, col = 1)
    fig.update_xaxes(title_text = "x")
    fig.update_yaxes(title_text = "u")
    utest = np.copy(uinit)
    for i in range(Nt):
        func(utest, *param)
    if plot_dim == None:
        m = min(np.min(utest), np.min(uinit))
        M = max(np.max(utest), np.max(uinit))
    else:
        m = min(np.min(utest[plot_dim]),np.min(uinit[plot_dim]))
        M = max(np.max(utest[plot_dim]), np.max(uinit[plot_dim]))
    dilat = (M - m) * dilat_size
    fig.update_layout(yaxis_range=[m - dilat , M + dilat ])

    ## subplot 2 #######
    fig.add_trace(my_surface(evo, X, Y, 0, n, dx), row = 1, col = 2)

    ## Evolving the frames ###############################################
    my_frames = [go.Frame(data = [my_scatter(x, func, u, plot_dim, *param),
                                  my_surface(evo, X, Y, i, n, dx)],
                traces=[0,1], name=str(i), layout_yaxis_range=[m , M])
                for i in range(Nt)]

    fig.frames = my_frames

    _fig_update_layout(fig, duration, title)

    fig.show()



def _fig_update_layout(fig, duration, title):
    ## Setting dello slider ##
    def frame_args(duration):
        return dict(frame={"duration": duration}, mode ="immediate", fromcurrent=True,
                    transition={"duration": duration,
                                "easing": "linear"},)
    # Definire i parametri dello slider
    sliders = [{"pad": {"b": 10, "t": 60}, "len": 0.5, "x": 0.1, "y": 0,
                "steps": [{"args": [[f.name], frame_args(duration)],
                           "label": str(k),
                           "method": "animate",}
                    for k, f in enumerate(fig.frames)],}]

    ## Layout generale della figura ##
    fig.update_layout( title=title, width=1200, height=700,
         scene=dict( xaxis_title='x', yaxis_title='t', zaxis_title='u', aspectratio=dict(x=1, y=1, z=1)),
         updatemenus = [
                {
                "buttons": [
                    {"args": [None, frame_args(duration)], "label": "&#9654;", # play symbol
                     "method": "animate",},
                    {"args": [[None], frame_args(duration)], "label": "&#9724;", # pause symbol
                     "method": "animate",
                    },],
                "direction": "left", "pad": {"r": 10, "t": 70}, "type": "buttons", "x": 0.1,"y": 0,
            }
         ], sliders=sliders
    )
    return
