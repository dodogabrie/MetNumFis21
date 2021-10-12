import plotly.graph_objects as go
import numpy as np
N = 100
nb_frames = 100
duration = 100

x = np.linspace(0, 2*np.pi, N)
def sin(x, phi):
    return np.sin(x + phi)
my_frames = [go.Frame(data = [go.Scatter(x = x, y = sin(x, 2*np.pi*i/nb_frames), mode = "lines")], name=str(i)) for i in range(nb_frames)]

fig = go.Figure(
        frames=my_frames)

# Add data to be displayed before animation starts
fig.add_trace(go.Scatter(
              x=x,
              y=sin(x, 0),
              mode='lines',)
            )

def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(duration)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]

# Layout
fig.update_layout(
         title='Simulazione con transiente',
         width=1200,
         height=700,
         scene=dict(
                    zaxis=dict(range=[-0.1, 6.8], autorange=False),
                    aspectratio=dict(x=1, y=1, z=1),
                    ),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(duration)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(duration)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
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

