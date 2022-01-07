#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sys
from sklearn.metrics.pairwise import cosine_distances
path="set path data"
path_write="path to write"
chunksize=50000


def convert_to_sparse_pandas(df, exclude_columns=[]):
    """
    Converts columns of a data frame into SparseArrays and returns the data frame with transformed columns.
    Use exclude_columns to specify columns to be excluded from transformation.
    :param df: pandas data frame
    :param exclude_columns: list
        Columns not be converted to sparse
    :return: pandas data frame
    """
    exclude_columns = set(exclude_columns)

    for (columnName, columnData) in df.iteritems():
        if columnName in exclude_columns:
            continue
        df[columnName] = pd.SparseArray(columnData.values, dtype='float64')

    return df

def main(*kwargs):
    chunks=pd.read_csv(path,index_col="n_cliente",converters={'n_cliente' : str}, chunksize=chunksize)
    df = pd.concat( [ convert_to_sparse_pandas(chunk,exclude_columns=["n_cliente","Cluster_Cuantitativo"]) for chunk in chunks ] )
    cluster_group = df.groupby("Cluster_Cuantitativo")
    for cluster, rows in cluster_group:
        print(cluster)
        print(rows.shape)
        df_cluster=rows.drop(['Cluster_Cuantitativo'], axis=1)
        b=pd.DataFrame(cosine_distances(df_cluster),index=df_cluster.index,columns=df_cluster.index)
        b.to_csv(path_write+cluster+".csv")




if __name__ == "__main__":
    main(sys.argv)



