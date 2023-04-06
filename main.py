import sys
import pandas as pd
import GroupMaxRank as gmr
from pathlib import Path

from gGraph import g_graph_estimation


if __name__ == "__main__":
    datafile = Path(sys.argv[1])
    datafolder = datafile.parent

    data = pd.read_csv(datafile, index_col=[0])
    skyline = pd.read_csv(datafolder / "skyline.csv", index_col=[0])
    skyline_index = skyline.index
    cells = pd.read_csv(datafolder/"cellsout.csv", index_col=[0])
    maxrank = pd.read_csv(datafolder/"maxrank.csv", index_col=[0])

    print("Loaded {} records from {}\n".format(data.shape[0], datafolder))

    if data.shape[1] == 2:
        gmr_cells = gmr.g_maxrank_2D(data, skyline_index)
        group_maxrank = min([gmr_cells[i].rank for i in range(len(gmr_cells))])
        for i in range(len(gmr_cells)):
            if gmr_cells[i].rank == group_maxrank:
                group_maxrank_q0 = gmr_cells[i].start
                break
        print("The group maxrank is {} obtained with query ({}, {})".format(group_maxrank, group_maxrank_q0, 1 - group_maxrank_q0))

    else:
        group_maxrank, gmr_cells = gmr.g_maxrank_HD(data, skyline_index)
        group_maxrank_q0 = list(gmr_cells[0].feasible_pnt.coord)
        print("{}, {}".format(maxrank, list(gmr_cells[0].feasible_pnt.coord)))

    data = data.rename(columns={
        data.columns[0]: "x",
        data.columns[1]: "y"
    })

    skyline = skyline.join(maxrank)
    skyline = skyline.join(cells)
    skyline['maxrank'] = skyline['maxrank'] + 1

    k_incr = int(sys.argv[2])

    g_graph_estimation(data, skyline, k_incr, group_maxrank, group_maxrank_q0)
