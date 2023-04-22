import random


def q_hat_estimation2D(data, sky):
    # All queries will be made using the higher maxrank (proper k_bar)
    k_bar = sky['maxrank'].max()
    q_hat = 0

    queries = []

    # Retrieved points of the skyline are removed, hence, the process continues until the list is empty
    while len(sky) > 0:
        k_bar_curr = sky['maxrank'].max()
        k_bar_curr_idx = sky.index[sky['maxrank'] == k_bar_curr]
        n_mincells = len(sky.loc[k_bar_curr_idx].drop_duplicates())

        # If the current point has multiple mincells, select one randomly
        if n_mincells > 1:
            mincell_idx = random.randint(0, n_mincells - 1)

            int_left = sky.loc[k_bar_curr_idx]['intLeft'].iloc[mincell_idx]
            int_right = sky.loc[k_bar_curr_idx]['intRight'].iloc[mincell_idx]
        else:
            int_left = sky.loc[k_bar_curr_idx]['intLeft'].iloc[0]
            int_right = sky.loc[k_bar_curr_idx]['intRight'].iloc[0]

        # The query paramenter is randomly picked within the cell;
        # this may have an impact in cells after the first
        q0 = random.uniform(int_left, int_right)

        # Execute the query
        topk_bar = topk2D(k_bar, data, q0, verbose=False)
        q_hat += 1

        # Filter the result for not yet retrieved skyline points
        sky_found = sky.index[[sky.index[i] in topk_bar.index for i in range(len(sky.index))]]
        sky_found = sky_found.drop_duplicates()

        queries.append([k_bar_curr_idx[0], (int_left, int_right)])

        # Remove retrieved skyline points
        sky = sky.drop(sky_found)

    return q_hat, k_bar, queries


def q_hat_estimation_HD(data, sky):
    # All queries will be made using the higher maxrank (proper k_bar)
    k_bar = sky['maxrank'].max()
    q_hat = 0

    queries = []

    # Retrieved points of the skyline are removed, hence, the process continues until the list is empty
    while len(sky) > 0:
        k_bar_curr = sky['maxrank'].max()
        k_bar_curr_idx = sky.index[sky['maxrank'] == k_bar_curr]
        n_mincells = len(sky.loc[k_bar_curr_idx].drop_duplicates())

        # If the current point has multiple mincells, select one randomly
        if n_mincells > 1:
            mincell_idx = random.randint(0, n_mincells - 1)
            query = sky.loc[k_bar_curr_idx]['query_found'].iloc[mincell_idx]
        else:
            query = sky.loc[k_bar_curr_idx]['query_found'].iloc[0]

        # Execute the query
        topk_bar = topkHD(k_bar, data, query, verbose=False)
        q_hat += 1

        # Filter the result for not yet retrieved skyline points
        sky_found = sky.index[[sky.index[i] in topk_bar.index for i in range(len(sky.index))]]
        sky_found = sky_found.drop_duplicates()

        queries.append([k_bar_curr_idx[0], query])

        # Remove retrieved skyline points
        sky = sky.drop(sky_found)

    return q_hat, k_bar, queries


def q_estimation2D(k, cell, data, sky):
    q = 0
    queries = []

    # First iteration, querying inside cell
    int_left = cell[0]
    int_right = cell[1]
    q0 = random.uniform(int_left, int_right)
    queries.append(q0)

    topk_res = topk2D(k, data, q0)
    q += 1

    sky_found = sky.index[[sky.index[i] in topk_res.index for i in range(len(sky.index))]]
    sky_found = sky_found.drop_duplicates()
    sky = sky.drop(sky_found)

    # Successive iterations, follow higher maxrank
    while len(sky) > 0:
        k_bar_curr = sky['maxrank'].max()
        k_bar_curr_idx = sky.index[sky['maxrank'] == k_bar_curr]
        n_mincells = len(sky.loc[k_bar_curr_idx].drop_duplicates())

        # If the current point has multiple mincells, select one randomly
        if n_mincells > 1:
            mincell_idx = random.randint(0, n_mincells - 1)

            int_left = sky.loc[k_bar_curr_idx]['intLeft'].iloc[mincell_idx]
            int_right = sky.loc[k_bar_curr_idx]['intRight'].iloc[mincell_idx]
        else:
            int_left = sky.loc[k_bar_curr_idx]['intLeft'].iloc[0]
            int_right = sky.loc[k_bar_curr_idx]['intRight'].iloc[0]

        # The query paramenter is randomly picked within the cell;
        # this may have an impact in cells after the first
        q0 = random.uniform(int_left, int_right)
        queries.append(q0)

        # Execute the query
        topk_res = topk2D(k, data, q0)
        q += 1

        # Filter the result for not yet retrieved skyline points
        sky_found = sky.index[[sky.index[i] in topk_res.index for i in range(len(sky.index))]]
        sky_found = sky_found.drop_duplicates()

        # Remove retrieved skyline points
        sky = sky.drop(sky_found)

    return q, queries


def q_estimationHD(k, query, data, sky):
    q = 0
    queries = []

    # First iteration, querying inside cell
    queries.append(query)

    topk_res = topkHD(k, data, query)
    q += 1

    sky_found = sky.index[[sky.index[i] in topk_res.index for i in range(len(sky.index))]]
    sky_found = sky_found.drop_duplicates()
    sky = sky.drop(sky_found)

    # Successive iterations, follow higher maxrank
    while len(sky) > 0:
        k_bar_curr = sky['maxrank'].max()
        k_bar_curr_idx = sky.index[sky['maxrank'] == k_bar_curr]
        n_mincells = len(sky.loc[k_bar_curr_idx].drop_duplicates())

        if n_mincells > 1:
            mincell_idx = random.randint(0, n_mincells - 1)
            query = sky.loc[k_bar_curr_idx]['query_found'].iloc[mincell_idx]
        else:
            query = sky.loc[k_bar_curr_idx]['query_found'].iloc[0]

        # The query paramenter is randomly picked within the cell;
        # this may have an impact in cells after the first
        queries.append(query)

        # Execute the query
        topk_res = topkHD(k, data, query)
        q += 1

        # Filter the result for not yet retrieved skyline points
        sky_found = sky.index[[sky.index[i] in topk_res.index for i in range(len(sky.index))]]
        sky_found = sky_found.drop_duplicates()

        # Remove retrieved skyline points
        sky = sky.drop(sky_found)

    return q, queries


def topk2D(k, data, q0, verbose=False):
    data['S(r)'] = q0 * data['x'] + (1 - q0) * data['y']
    data = data.sort_values(by=['S(r)'], ascending=True)

    if verbose:
        print("Executed top{} query with params=[{}, {}]".format(k, q0, 1 - q0))

    return data[:k]


def topkHD(k, data, query, verbose=False):
    query = get_array_from_string(query)
    temp = 0

    for i in range(len(query)):
        temp += query[i] * data['x' + str(i + 1)]

    data['S(r)'] = temp
    data = data.sort_values(by=['S(r)'], ascending=True)

    if verbose:
        print("Executed top{} query with params=[{}]".format(k, query))

    return data[:k]

def get_array_from_string(string):
    string = string.replace("[", "")
    string = string.replace("]", "")
    string = string.replace(" ", "")
    string = string.split(",")
    return [float(x) for x in string]
