# # trajectory_plot_multiple.py
# from bokeh.plotting import figure, curdoc
# from bokeh.models import Slider, ColumnDataSource, Range1d
# from bokeh.layouts import column, layout
# import numpy as np

# # Function to create a sample trajectory
# def create_trajectory(omega):
#     time = np.linspace(0, 10, 100)
#     x = np.cos(omega * time)
#     y = np.sin(omega * time)
#     return time, x, y

# # Generate three different trajectories
# time1, x1, y1 = create_trajectory(1)  # Normal frequency
# time2, x2, y2 = create_trajectory(2)  # Double frequency
# time3, x3, y3 = create_trajectory(3)  # Triple frequency

# # Create a ColumnDataSource for each trajectory
# source1 = ColumnDataSource(data={'time': time1, 'x': x1, 'y': y1})
# source2 = ColumnDataSource(data={'time': time2, 'x': x2, 'y': y2})
# source3 = ColumnDataSource(data={'time': time3, 'x': x3, 'y': y3})

# # Create three plots
# p1 = figure(width=400, height=400, title="Trajectory 1", x_axis_label='X', y_axis_label='Y',
#             x_range=Range1d(-1.2, 1.2), y_range=Range1d(-1.2, 1.2))
# p1.scatter('x', 'y', source=source1, size=8, color="navy", alpha=0.6)

# p2 = figure(width=400, height=400, title="Trajectory 2", x_axis_label='X', y_axis_label='Y',
#             x_range=Range1d(-1.2, 1.2), y_range=Range1d(-1.2, 1.2))
# p2.scatter('x', 'y', source=source2, size=8, color="red", alpha=0.6)

# p3 = figure(width=400, height=400, title="Trajectory 3", x_axis_label='X', y_axis_label='Y',
#             x_range=Range1d(-1.2, 1.2), y_range=Range1d(-1.2, 1.2))
# p3.scatter('x', 'y', source=source3, size=8, color="green", alpha=0.6)

# # Update function that updates all plots
# def update_plot(attr, old, new):
#     t = time_slider.value
#     indices1 = time1 <= t
#     indices2 = time2 <= t
#     indices3 = time3 <= t
#     source1.data = {'time': time1[indices1], 'x': x1[indices1], 'y': y1[indices1]}
#     source2.data = {'time': time2[indices2], 'x': x2[indices2], 'y': y2[indices2]}
#     source3.data = {'time': time3[indices3], 'x': x3[indices3], 'y': y3[indices3]}

# # Create a slider to control the time, affecting all plots
# time_slider = Slider(start=0, end=10, value=1, step=0.1, title="Time")
# time_slider.on_change('value', update_plot)

# # Layout setup
# # layout = column(p1, p2, p3, time_slider)

# # Add to document
# curdoc().add_root(layout([[p1, p2, p3], time_slider]))
# curdoc().title = "Multiple Trajectory Viewer"



# from bokeh.models import ColumnDataSource, Slider, Button, TabPanel, Tabs, CustomJS, Range1d
# from bokeh.layouts import column, row
# from bokeh.plotting import figure, curdoc
# import numpy as np

# # Global dictionary to keep track of selected tabs
# selected_tabs = {}

# def create_trajectory(omega, phase=0):
#     time = np.linspace(0, 10, 100)
#     x = np.cos(omega * time + phase)
#     y = np.sin(omega * time + phase)
#     return time, x, y

# def generate_plots(omega):
#     time, x, y = create_trajectory(omega)
#     source = ColumnDataSource(data={'time': time, 'x': x, 'y': y, 'visible_x': x, 'visible_y': y})
#     p = figure(width=300, height=300, title=f"Trajectory ω={omega}", x_axis_label='X', y_axis_label='Y', x_range=Range1d(-1.2, 1.2), y_range=Range1d(-1.2, 1.2))
#     p.scatter('visible_x', 'visible_y', source=source, size=8)
#     return p, source

# def setup_tab(i):
#     p1, source1 = generate_plots(i+1)
#     p2, source2 = generate_plots(i+1.5)
#     p3, source3 = generate_plots(i+2)
#     sources = [source1, source2, source3]

#     def update_plots(attr, old, new):
#         for source in sources:
#             t = new
#             data = source.data
#             visible_indices = data['time'] <= t
#             full_length = len(data['time'])
#             new_visible_x = np.full(full_length, np.nan)
#             new_visible_y = np.full(full_length, np.nan)
#             new_visible_x[visible_indices] = data['x'][visible_indices]
#             new_visible_y[visible_indices] = data['y'][visible_indices]
#             source.data.update({'visible_x': new_visible_x, 'visible_y': new_visible_y})

#     time_slider = Slider(start=0, end=10, value=10, step=0.1, title="Time")
#     time_slider.on_change('value', update_plots)

#     button = Button(label="Deselect (Exclude from Averaging)", button_type="success", width=200, css_classes=["selected-button"])
#     selected_tabs[i] = True  # Initialize as selected by default

#     def toggle_selection():
#         selected_tabs[i] = not selected_tabs[i]
#         if selected_tabs[i]:
#             button.label = "Deselect (Exclude from Averaging)"
#             button.css_classes = ["selected-button"]
#         else:
#             button.label = "Select (Include in Averaging)"
#             button.css_classes = ["default-button"]
#         print(f"Currently selected tabs: {[k for k, v in selected_tabs.items() if v]}")

#     button.on_click(toggle_selection)

#     update_plots('value', None, time_slider.value)

#     tab_content = column(row(p1, p2, p3), time_slider, button)
#     tab = TabPanel(child=tab_content, title=f"Tab ω={i+1}")
#     return tab

# # Setting up tabs
# tabs = [setup_tab(i) for i in range(3)]
# tabs_widget = Tabs(tabs=tabs)

# curdoc().add_root(tabs_widget)
# curdoc().title = "Interactive Trajectory Analysis with Averaging"

# print("Initial selection of tabs: ", [k for k, v in selected_tabs.items() if v])



from bokeh.models import ColumnDataSource, Slider, Button, TabPanel, Tabs, Range1d
from bokeh.layouts import layout
from bokeh.plotting import figure, curdoc
import numpy as np



def create_trajectory(omega, phase=0):
    time = np.linspace(0, 10, 100)
    x = np.cos(omega * time + phase)
    y = np.sin(omega * time + phase)
    return time, x, y


def generate_plots(omega):
    time, x, y = create_trajectory(omega)
    source = ColumnDataSource(data={'time': time, 'x': x, 'y': y, 'visible_x': x, 'visible_y': y})
    p = figure(width=300, height=300, title=f"Trajectory ω={omega}", x_axis_label='X', y_axis_label='Y', x_range=Range1d(-1.2, 1.2), y_range=Range1d(-1.2, 1.2))
    p.scatter('visible_x', 'visible_y', source=source, size=8)
    return p, source


def update_visibility(source, new_value):
    data = source.data.copy()
    visible_indices = data['time'] <= new_value
    data['visible_x'] = np.where(visible_indices, data['x'], np.nan)
    data['visible_y'] = np.where(visible_indices, data['y'], np.nan)
    source.data = data


def create_averaged_sources():
    time = np.linspace(0, 10, 100)  # This ensures time is consistently 100 elements long
    initial_data = {'time': time, 'x': np.array([]), 'y': np.array([]), 'visible_x': np.array([]), 'visible_y': np.array([])}
    # Initially setting x, y, visible_x, and visible_y to empty arrays of length matching 'time'
    return [ColumnDataSource(data={key: np.full_like(time, np.nan) if key != 'time' else time for key in initial_data}) for _ in range(3)]


## Update visibility based on the averaged time slider
def update_visibility_for_averaged():
    global current_slider_value
    for source in averaged_sources:
        
        ## Extract current data to variables
        time = source.data['time'].copy()
        x = source.data['x'].copy()
        y = source.data['y'].copy()

        ## Calculate visibility based on the current slider value
        visible_indices = time <= current_slider_value
        visible_x = np.where(visible_indices, x, np.nan)  # Create new arrays for visibility
        visible_y = np.where(visible_indices, y, np.nan)

        ## Construct a new dictionary for the data source update
        new_data = {
            'time': time,
            'x': x,
            'y': y,
            'visible_x': visible_x,
            'visible_y': visible_y
        }

        ## Assign this new dictionary to the data source
        source.data = new_data


## Ensure update functions handle the full and empty cases properly
def update_averaged_plots(plot_sources, averaged_sources):
    for i_plot, sources in enumerate(plot_sources):

        xs, ys = [], []

        for i_tab in range(3):
            if selected_tabs.get(i_tab, False):  # Check if the tab is selected

                xs.append(sources[i_tab].data['x'])
                ys.append(sources[i_tab].data['y'])

        if xs:
            # Calculate the average for visible data
            time = sources[0].data['time']  # Assuming all sources in a tab share the same time array
            averaged_sources[i_plot].data.update({
                'time': time,
                'x': np.mean(xs, axis=0), 'y': np.mean(ys, axis=0),
                'visible_x': np.mean(xs, axis=0), 'visible_y': np.mean(ys, axis=0)
            })
        else:
            # Clear the plot if the tab is not selected
            time = sources[0].data['time']
            null_data = np.full_like(time, np.nan)
            averaged_sources[i_plot].data.update({
                'time': time,
                'x': null_data, 'y': null_data,
                'visible_x': null_data, 'visible_y': null_data
            })

    update_visibility_for_averaged()  # Ensure visibility is updated according to the current slider value


def slider_update(attr, old, new):
    global current_slider_value
    current_slider_value = new  # Update the global variable whenever the slider changes
    update_visibility_for_averaged()


def setup_tab(i_tab):
    plots = []
    sources = []
    for i_plot in range(3):
        p, source = generate_plots(i_tab + i_plot * 0.5)
        plots.append(p)

        sources.append(source)
        plot_sources[i_plot].append(source)
        # selected_tabs[source.id] = True

    selected_tabs[i_tab] = True

    time_slider = Slider(start=0, end=10, value=10, step=0.1, title="Time")
    time_slider.on_change('value', lambda attr, old, new: [update_visibility(src, new) for src in sources])

    button = Button(label="Deselect (Exclude from Averaging)", button_type="success", width=200)

    def toggle_selection():
        # for src in sources:
        #     current_selection = not selected_tabs.get(src.id, True)
        #     selected_tabs[src.id] = current_selection
        selected_tabs[i_tab] = not selected_tabs[i_tab]

        button.label = "Deselect (Exclude from Averaging)" if selected_tabs[i_tab] else "Select (Include in Averaging)"
        update_averaged_plots(plot_sources, averaged_sources)
        print_selected_tabs()

    button.on_click(toggle_selection)
    update_averaged_plots(plot_sources, averaged_sources)

    tab_content = layout([plots, [time_slider], [button]])
    return TabPanel(child=tab_content, title=f"Tab ω={i_tab + 1}")


def print_selected_tabs():

    selected = [f"Tab {index + 1}" for index, is_selected in selected_tabs.items() if is_selected]
    print("Currently selected tabs:", selected if selected else "No tabs selected")



selected_tabs = {}  # Tracks which tabs are included in the averaging
current_slider_value = 10

## Global variables for plots and averaged sources
plot_sources = [[], [], []]
averaged_sources = create_averaged_sources()
averaged_plots = [figure(width=300, height=300, x_axis_label='X', y_axis_label='Y', x_range=Range1d(-1.2, 1.2), y_range=Range1d(-1.2, 1.2), title=f"Averaged Plot {chr(65+i)}") for i in range(3)]
for plot, source in zip(averaged_plots, averaged_sources):
    plot.scatter('visible_x', 'visible_y', source=source, size=8)

## Time slider for the averaged trajectory plots
averaged_time_slider = Slider(start=0, end=10, value=10, step=0.1, title="Max Time for Averaged Data")
averaged_time_slider.on_change('value', slider_update)

tabs = [setup_tab(i) for i in range(3)]
tabs_widget = Tabs(tabs=tabs)

layout = layout([[tabs_widget], averaged_plots, [averaged_time_slider]])
curdoc().add_root(layout)
curdoc().title = "Interactive Trajectory Analysis with Averaging"

print_selected_tabs()
