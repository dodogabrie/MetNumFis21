import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def animated_basic(x, uinit, func, Nt, *param):
    def animate(i):
        # LAX goes directly here
        line.set_ydata( func(u, *param) )  # update the data
        return line,
    
    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    u = np.copy(uinit)
    fig, ax1 = plt.subplots()
    line, = ax1.plot(x, uinit)
    ani = animation.FuncAnimation(fig, animate, np.arange(0, Nt), init_func=init,
                                  interval=25, blit=True)
    plt.show()


def animated_full(func, func_mesh, x, t, uinit, *param, imported_title = None):
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

    ######### Definire un subplot con Plolty
    fig = make_subplots( rows=1, cols=2, 
                         subplot_titles=('Title1', 'Title2'),
                         horizontal_spacing=0.051,
                         specs=[[{"type": "scatter"} , {"type": "surface"}]]
    )
    ## subplot 1 #######
    init_scatter = go.Scatter( x=x, y=uinit, mode='lines')
    fig.add_trace(init_scatter, row = 1, col = 1)
    m = np.min(uinit)
    M = np.max(uinit)
    fig.update_layout(yaxis_range=[m , M ])
 
    ## subplot 2 #######
    def my_surface(evo, X, Y, i, n, dx):
        startt = i * dt
        endt = (i+1) * dt
        contoursy = {"show" : True, "start": startt, "end": endt, "size": 2*dt, "width" : 3}
        return go.Surface( z=evo, x=X, y=Y, opacity = 0.3, contours = {'y': contoursy})

    fig.add_trace(my_surface(evo, X, Y, 0, n, dx), row = 1, col = 2)

    ## Animazione #####
    my_frames = [go.Frame(data = [go.Scatter(x = x, y = func(u, *param), mode = "lines"),
                                  my_surface(evo, X, Y, i, n, dx)], traces=[0,1], name=str(i), layout_yaxis_range=[m , M]) for i in range(Nt)]
    fig.frames = my_frames
    def frame_args(duration):
        return dict(frame={"duration": duration},
                    mode ="immediate",
                    fromcurrent=True,
                    transition={"duration": duration, "easing": "linear"},
                    )

    
    sliders = [{"pad": {"b": 10, "t": 60},
                "len": 0.5,
                "x": 0.1,
                "y": 0,
                "steps": [{"args": [[f.name], frame_args(duration)],
                           "label": str(k),
                           "method": "animate",}
                    for k, f in enumerate(fig.frames)],}]

    # Layout
    fig.update_layout(
         title=imported_title,
         width=1200,
         height=700,
         scene=dict(aspectratio=dict(x=1, y=1, z=1)),
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
