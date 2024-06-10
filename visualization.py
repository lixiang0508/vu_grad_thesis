from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from Obstacle import Obstacle

terminals = [(0.6, 0.6), (0.65, 0.65), (0.7, 0.6)]  # Example terminals
steinerpts = [(0.55, 0.55), (0.6, 0.5)]  # Example Steiner points


# Function to create polygons for obstacles
def create_polygon(coords, color, ax):
    polygon = Polygon(coords, closed=True, facecolor=color, edgecolor='black', alpha=0.5)
    ax.add_patch(polygon)


bins = "110"  # Example binary string for obstacle corner inclusion
soft_obstacles = [
    Obstacle(1.1, 'soft', [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)]),
    Obstacle(1.1, 'soft', [(0.15, 0.15), (0.25, 0.15), (0.25, 0.25), (0.15, 0.25)])
]
hard_obstacles = [
    Obstacle(1.1, 'hard', [(0.3, 0.1), (0.4, 0.1), (0.4, 0.2), (0.3, 0.2)]),
    Obstacle(1.1, 'hard', [(0.35, 0.15), (0.45, 0.15), (0.45, 0.25), (0.35, 0.25)])
]
mst_edges = [((0.55, 0.55), (0.6, 0.5)), ((0.6, 0.5), (0.65, 0.65))]  # Example edges of the MST


def vis(terminals, steiner_points, corners, soft_obstacles, hard_obstacles, steiner_tree_edges, path):
    # Now let's modify the plot code to handle these objects
    '''
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot soft obstacles
    for polygon_coords in soft_obstacles:
        create_polygon(polygon_coords, 'lightgreen', ax)

    # Plot hard obstacles
    for polygon_coords in hard_obstacles:
        create_polygon(polygon_coords, 'orange', ax)

    # Plot terminals
    terminal_plot = ax.plot(*zip(*terminals), 'o', color='red', label='Terminals', markersize=12)[0]

    # Plot Steiner points
    steiner_point_plot = ax.plot(*zip(*steiner_points), 'o', color='blue', label='Steiner Points', markersize=12)[0]
    corners_plot = ax.plot(*zip(*corners), 'o', color='black', label='Obstacle Corners', markersize=12)[0]

    # Plot Steiner tree edges
    for edge in steiner_tree_edges:
        line = Line2D([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color='black')
        ax.add_line(line)

    # Creating custom legends for the polygons as they are not directly supported
    soft_obstacle_legend = Line2D([0], [0], marker='o', color='w', label='Soft Obstacle',
                                  markerfacecolor='lightgreen', markersize=15)
    hard_obstacle_legend = Line2D([0], [0], marker='o', color='w', label='Hard Obstacle',
                                  markerfacecolor='orange', markersize=15)
    terminal_plot_legend = Line2D([0], [0], marker='o', color='w', label='Terminal',
                                  markerfacecolor='red', markersize=15)
    steiner_point_plot_legend = Line2D([0], [0], marker='o', color='w', label='Steiner point',
                                       markerfacecolor='blue', markersize=15)
    corners_plot_legend = Line2D([0], [0], marker='o', color='w', label='Obstacle corner',
                                 markerfacecolor='black', markersize=15)
    steiner_edge_legend = Line2D([0], [0], color='black', lw=2, label='Steiner Tree Edge')

    # Add legends
    ax.legend(handles=[soft_obstacle_legend, hard_obstacle_legend, terminal_plot_legend, steiner_point_plot_legend,
                       corners_plot_legend, steiner_edge_legend], loc='lower right',
              prop={'size': 18, 'weight': 'bold'},
              bbox_to_anchor=(1, 1))

    # Set the plot boundaries

    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # Show the plot with a legend
    plt.xlabel('GA '+path)
    plt.show()
    '''
    print('plot')
