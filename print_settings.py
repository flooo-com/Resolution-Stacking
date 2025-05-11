import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update(mpl.rcParamsDefault)


# Update Matplotlib rcParams
mpl.rcParams.update({
    'font.family': 'Calibri',  # Font family for all text
    #'figure.dpi': 160,  # High resolution for figures
    
    # Axes settings
    'axes.titlesize': 12,  # Font size for the axes title
    'axes.titleweight': 'normal',  # Weight for the axes title
    'axes.labelsize': 10,  # Font size for the axes labels
    
    # Tick settings
    # 'xtick.direction': 'in',  # Ticks pointing inwards
    # 'ytick.direction': 'in',  # Ticks pointing inwards
    'xtick.minor.visible': False,  # Show minor ticks on x-axis
    'ytick.minor.visible': False,  # Show minor ticks on y-axis
    # 'xtick.major.visible': False,  # Show major ticks on x-axis
    # 'ytick.major.visible': False,  # Show major ticks on y-axis

    'xtick.major.size': 3,  # Length of major ticks on x-axis
    'xtick.minor.size': 3,  # Length of minor ticks on x-axis
    'ytick.major.size': 3,  # Length of major ticks on y-axis
    'ytick.minor.size': 3,  # Length of minor ticks on y-axis

    # Grid settings
    # 'axes.grid': True,  # Enable grid by default
    # 'axes.grid.which': 'both',  # Show both major and minor grids
    # 'grid.linestyle': ':',  # Dotted grid lines
    # 'grid.alpha': 0.3,  # Transparency for major grid lines
    # 'grid.linewidth': 0.6,  # Line width for major grid lines

})

plt.rcParams['ytick.right'] = plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
