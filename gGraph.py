import matplotlib.pyplot as plt

from qHat import q_hat_estimation2D, q_estimation2D, topk2D, q_hat_estimation_HD, q_estimationHD, topkHD
from sklearn import metrics


def g_graph_estimation_2D(data, sky, k_incr, group_maxrank, group_maxrank_query):
    # points of the graph
    g_points = []

    print("Estimating q_hat...")
    # estimate of the point in the "top-left corner"
    q_hat, k_bar, queries = q_hat_estimation2D(data, sky)
    print("The dataset has q_hat={} and k_bar={}\n".format(q_hat, k_bar))

    k = k_bar
    q = q_hat
    targets = [str(target) for target, cell in queries]
    targets.append("multi")
    print("Estimating k_hat...")
    print("k is set to increment in steps of {}".format(k_incr))

    while q > 1:
        for i in range(len(queries)):
            target_point, cell = queries[i]

            q_i, queries_i = q_estimation2D(k, cell, data, sky)
            if q_i < q:
                q = q_i
                to_append = (k, q, str(target_point))
            elif q_i == q:
                to_append = (k, q, "multi")

        if k >= group_maxrank and len(g_points) >= 1:
            break
        # add to the list k, q and id of the point
        g_points.append(to_append)
        k += k_incr

    # Add the point in the "bottom-right" corner
    last_point = (group_maxrank, 1, str(-1))
    g_points.append(last_point)
    targets.append(str(g_points[-1][2]))

    k_hat_query = topk2D(group_maxrank, data, group_maxrank_query)

    # ----------------- PLOTTING AND AREAS COMPUTATION -----------------
    fig, axs = plt.subplots(1, 2)

    fig.set_figwidth(16)
    fig.set_figheight(8)

    print("The dataset has q_bar={} and k_hat={}".format(g_points[-1][1], group_maxrank))
    print_graph2D(axs[0], g_points, targets)
    print_k_hat_query(axs[1], data, sky, k_hat_query)

    compute_areas(g_points)

    plt.show()


def g_graph_estimation_HD(data, sky, k_incr, group_maxrank):
    # points of the graph
    g_points = []

    print("Estimating q_hat...")
    # estimate of the point in the "top-left corner"
    q_hat, k_bar, queries = q_hat_estimation_HD(data, sky)
    print("The dataset has q_hat={} and k_bar={}\n".format(q_hat, k_bar))

    k = k_bar
    q = q_hat
    targets = [str(target) for target, query in queries]
    targets.append("multi")
    print("Estimating k_hat...")
    print("k is set to increment in steps of {}".format(k_incr))

    while q > 1:
        for i in range(len(queries)):
            target_point, cell = queries[i]

            q_i, queries_i = q_estimationHD(k, cell, data, sky)
            if q_i < q:
                q = q_i
                to_append = (k, q, str(target_point))
            elif q_i == q:
                to_append = (k, q, "multi")
            else:  # TODO: valutare con q_i invece di q
                to_append = (k, q, "multi")

        if k >= group_maxrank and len(g_points) >= 1:
            break
        # add to the list k, q and id of the point
        g_points.append(to_append)
        k += k_incr

    last_point = (group_maxrank, 1, str(-1))
    g_points.append(last_point)
    targets.append(str(g_points[-1][2]))

    print("The dataset has q_bar={} and k_hat={}".format(g_points[-1][1], group_maxrank))

    # ----------------- PLOTTING AND AREAS COMPUTATION -----------------
    print_graphHD(g_points, targets)

    compute_areas(g_points)

    plt.show()


def g_graph_estimation_HD_without_group_maxrank(data, sky, k_incr):
    # points of the graph
    g_points = []

    print("Estimating q_hat...")
    # estimate of the point in the "top-left corner"
    q_hat, k_bar, queries = q_hat_estimation_HD(data, sky)
    print("The dataset has q_hat={} and k_bar={}\n".format(q_hat, k_bar))

    k = k_bar
    q = q_hat
    targets = [str(target) for target, query in queries]
    targets.append("multi")
    print("Estimating k_hat...")
    print("k is set to increment in steps of {}".format(k_incr))

    while q > 1:
        for i in range(len(queries)):
            target_point, cell = queries[i]

            q_i, queries_i = q_estimationHD(k, cell, data, sky)
            if q_i < q:
                q = q_i
                to_append = (k, q, str(target_point))
            elif q_i == q:
                to_append = (k, q, "multi")
            else:  # TODO: valutare con q_i invece di q
                to_append = (k, q, "multi")

        # add to the list k, q and id of the point
        g_points.append(to_append)
        k += k_incr

    print("The dataset has q_bar={} and k_hat={}".format(g_points[-1][1], g_points[-1][0]))

    # ----------------- PLOTTING AND AREAS COMPUTATION -----------------
    print_graphHD(g_points, targets)

    compute_areas(g_points)

    plt.show()


def print_graph2D(ax, points, targets):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    coloring = 0

    for t in targets:
        k = [k for k, q, target in points if target == t]
        q = [q for k, q, target in points if target == t]

        if t == "multi":
            ax.scatter(k, q, marker='o', color="grey", label="Multiple Targets")
        else:
            ax.scatter(k, q, marker='o', color=colors[coloring % len(colors)], label="Target Point " + t)
            coloring += 1

    ax.grid()

    ax.set_xlabel("k")
    ax.set_ylabel("q")

    k = [k for k, q, target in points]
    q = [q for k, q, target in points]
    ax.set_yticks(range(1, max(q) + 1, 1))

    ax.annotate(r"$\underbar \!\!\! k = {}$".format(k[0]), (k[0], max(q)))
    ax.annotate(r"$\bar k = {}$".format(k[-1]), (k[-1], 1))

    ax.legend(loc="best")

    ax.set_title("g Graph estimated using Maxrank")


def print_graphHD(points, targets):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    coloring = 0

    for t in targets:
        k = [k for k, q, target in points if target == t]
        q = [q for k, q, target in points if target == t]

        if t == "multi":
            plt.scatter(k, q, marker='o', color="grey", label="Multiple Targets")
        else:
            plt.scatter(k, q, marker='o', color=colors[coloring % len(colors)], label="Target Point " + t)
            coloring += 1

    plt.grid()

    plt.xlabel("k")
    plt.ylabel("q")

    k = [k for k, q, target in points]
    q = [q for k, q, target in points]
    plt.yticks(range(1, max(q) + 1, 1))

    plt.annotate(r"$\underbar \!\!\! k = {}$".format(k[0]), (k[0], max(q)))
    plt.annotate(r"$\bar k = {}$".format(k[-1]), (k[-1], 1))

    plt.legend(loc="best")

    plt.title("g Graph estimated using Maxrank")


def print_k_hat_query(ax, data, sky, query):
    data = data[~data.index.isin(query.index)]
    query = query[~query.index.isin(sky.index)]

    ax.scatter(data[data.columns[0]], data[data.columns[1]], marker='o', color="grey")
    ax.scatter(query[query.columns[0]], query[query.columns[1]], marker='o', color="red", label="Query Points")
    ax.scatter(sky[sky.columns[0]], sky[sky.columns[1]], marker='^', color="blue", label="Skyline Points")

    ax.set_xlabel(sky.columns[0])
    ax.set_ylabel(sky.columns[1])

    ax.legend(loc="best")

    ax.set_title("Single Query Retrival of Skyline")


def compute_areas(g_points):
    k_hat = g_points[-1][0]
    q_hat = g_points[0][1]
    k_bar = g_points[0][0]
    # side_rect is the rectangle with k in [1, k_bar] which should be added in area computation
    side_rect = (q_hat - 1) * (k_bar - 1)
    # base_rect is the rectangle with q in [0,1] which should be ignored in area computation
    base_rect = (k_hat - k_bar)  # * 1
    x_all = [k for k, q, target in g_points]
    y_all = [q for k, q, target in g_points]
    exact_area = metrics.auc(x_all, y_all) - base_rect + side_rect

    x_appr = [x_all[0], x_all[-1]]
    y_appr = [y_all[0], y_all[-1]]
    approx_area = metrics.auc(x_appr, y_appr) - base_rect + side_rect

    rect_area = (g_points[-1][0] - k_bar) * (y_all[0] - 1) + side_rect
    normalized_area = exact_area / rect_area

    print("The approximated area under the curve is {}".format(approx_area))
    print("The exact area under the curve is {}".format(exact_area))
    print("The rectangle area is {}".format(rect_area))
    print("The normalized area under the curve is {}".format(normalized_area))
