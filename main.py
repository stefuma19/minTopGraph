import sys
import pandas as pd
import GroupMaxRank as gmr
from pathlib import Path

from gGraph import g_graph_estimation_2D, g_graph_estimation_HD, g_graph_estimation_HD_without_group_maxrank

"""
Main
Run the MinTopGraph computation by calling : "python main.py path/to/datafile increment_of_k compute_group_max_rank" 
    where:
    datafile:   A CSV file containing all data points, correlated with an id. Data should be normalized beforehand, 
                see example folder for reference.
    increment_of_k:  The increment of k for the estimation of q(s) in the MinTopGraph algorithm.
    compute_group_max_rank: a boolean value that indicates if the GroupMaxRank should be computed or not. If not set
                              it is computed by default.
       
Note: the program in order to work needs other files in the same folder as the datafile. These files are:
    skyline.csv: the skyline of the dataset
    maxrank.csv: the maxrank of each skyline point in the dataset
    cellsout.csv: the mincells of each skyline point in the dataset in the case of 2D datasets
    cells.csv: the mincells of each skyline point in the dataset in the case of HD datasets                       
                              
The output of the computation consists in one or two graphs, depending on the number of dimensions of the dataset.
The MinTopGraph is always displayed, the query space and the skyline are displayed only if the dataset is 2D.
"""

if __name__ == "__main__":
    datafile = Path(sys.argv[1])
    datafolder = datafile.parent

    data = pd.read_csv(datafile, index_col=[0])
    skyline = pd.read_csv(datafolder / "skyline.csv", index_col=[0])
    skyline_index = skyline.index
    if data.shape[1] == 2:
        cells = pd.read_csv(datafolder / "cellsout.csv", index_col=[0])
    else:
        cellsHD = pd.read_csv(datafolder / "cells.csv", index_col=[0])
    maxrank = pd.read_csv(datafolder / "maxrank.csv", index_col=[0])
    k_incr = int(sys.argv[2])
    try:
        compute_group_max_rank = int(sys.argv[3])
    except IndexError:
        compute_group_max_rank = 1  # If not specified, compute the group max rank

    print("Loaded {} records from {}\n".format(data.shape[0], datafolder))

    sigma = len(skyline)
    skyline = skyline.join(maxrank)
    if data.shape[1] > 2:
        skyline = skyline.join(cellsHD)
    else:
        skyline = skyline.join(cells)
    skyline['maxrank'] = skyline['maxrank']# + 1
    # 1 should be added only if the maxrank is computed with Mouratidis' convention, i.e. min(maxrank) = 0

    # Computes the GroupMaxRank in 2 dimensions
    if data.shape[1] == 2:
        gmr_cells = gmr.g_maxrank_2D(data, skyline_index)
        group_maxrank = min([gmr_cells[i].rank for i in range(len(gmr_cells))])
        for i in range(len(gmr_cells)):
            if gmr_cells[i].rank == group_maxrank:
                group_maxrank_q0 = gmr_cells[i].start
                break
        if group_maxrank < len(skyline):
            group_maxrank = len(skyline)
        print("The group maxrank is {} obtained with query ({}, {})".format(group_maxrank, group_maxrank_q0,
                                                                            1 - group_maxrank_q0))
    # Computes the GroupMaxRank in more than 2 dimensions if the flag is set
    elif compute_group_max_rank == 1:
        group_maxrank, gmr_cells = gmr.g_maxrank_HD(data, skyline_index)
        group_maxrank_q0 = list(gmr_cells[0].feasible_pnt.coord)
        print("{}, {}".format(group_maxrank, list(gmr_cells[0].feasible_pnt.coord)))

    if data.shape[1] == 2:
        data = data.rename(columns={data.columns[0]: "x", data.columns[1]: "y"})
        g_graph_estimation_2D(data, skyline, k_incr, group_maxrank, group_maxrank_q0, sigma)
    elif compute_group_max_rank == 1:
        g_graph_estimation_HD(data, skyline, k_incr, group_maxrank, sigma)
    else:  # compute_group_max_rank == 0
        g_graph_estimation_HD_without_group_maxrank(data, skyline, k_incr, sigma)
