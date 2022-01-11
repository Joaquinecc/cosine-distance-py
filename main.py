import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix, vstack,csr_matrix
from scipy.sparse import lil_matrix
import numpy as np
import time
start_time = time.time()
BYTES_TO_MB_DIV = 0.000001

'''

Global Variables

'''
path="C:/Users/datoadmin/bristol/rating-productos.csv"
path_write="C:/Users/datoadmin/bristol/ranking-client.csv"
chunksize=10000


def print_memory_usage_of_data_frame(df):
    mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
    print("Memory usage is " + str(mem) + " MB")

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
        df[columnName] = pd.arrays.SparseArray(columnData.values, dtype='float64')

    return df
def calc_rank(cx,index,N=10):
    '''
    Ranking the top N client  cosine similarity of each client

    Params:
    N: uint : number of rank. Ej top 10, so n=10
    cx: spare matrix
    index: index of each row

    return:
    array with the following format
    [["current client","neighbor","cosine distance","rank position"],..]
    Ex:
    [['0000250', '0000250', 1.0000000000000002, 0],
     ['0000250', '015877', 0.718819103532109, 1],
     ['0000250', 'MO621484', 0.7160569503736789, 2],
     ['0000250', 'MO621430', 0.7160569503736789, 3],
     ['0000250', 'MO620178', 0.7160569503736789, 4],
     ['0000250', 'MO620138', 0.7160569503736789, 5],
     ['0000250', 'MO618804', 0.7160569503736789, 6],
     ['0000250', 'MO617581', 0.7160569503736789, 7],
     ['0000250', 'MO615424', 0.7160569503736789, 8],
     ['0000250', 'MO614825', 0.7160569503736789, 9],
     ['001384', '001384', 1.0, 0],
     ...]

    '''
    current_row=0
    data=[]
    temp=[]

    for i,j,v in zip(cx.row, cx.col, cx.data):
        if(i != current_row):
            #sort in descending order
            temp=sorted(temp, key=lambda d: d['cosine distance']) 
            
            #only rank the top N element
            size= N if N< len(temp) else len(temp) 
            [element.update({"rank":index+1}) for index,element in enumerate(temp[0:size]) ]
            for element in temp[0:size]:
                            #Current Client       Neighbor       cosine distance  rank position"
                data.append([index[current_row],element["Neighbor"],element["cosine distance"],element["rank"]])
            current_row=i
            temp=[]
                                        #cosine distance
        temp.append({"Neighbor":index[j],"cosine distance":1-v})
    return data


def main():
    print("Start..")
    print("--- %s seconds ---" % (time.time() - start_time))

    #Read csv file in chunks so memory not colapse
    chunks=pd.read_csv(path,index_col="n_cliente",converters={'n_cliente' : str}, chunksize=chunksize)
    df = pd.concat( [ convert_to_sparse_pandas(chunk,exclude_columns=["n_cliente","Cluster_Cuantitativo"]) for chunk in chunks ] )
    print_memory_usage_of_data_frame(df)
    print("--- Reading %s seconds ---" % (time.time() - start_time))

    cluster_group = df.groupby("Cluster_Cuantitativo")
    data=[]
    for cluster, rows in cluster_group:
        print(cluster)
        print(rows.shape)
        indexs=rows.index
        
        #Calc Cosine similarity
        sp_cl=csr_matrix(rows.drop(['Cluster_Cuantitativo'], axis=1))
        sp_cl=cosine_similarity(sp_cl,dense_output=False)
        sp_cl=coo_matrix(sp_cl)
        #calc rank data
        sp_cl=calc_rank(sp_cl,indexs,N=30)
        if(len(data) == 0):
            data=sp_cl.copy()
        else:
            data = np.concatenate((data, sp_cl))
        print("--- Reading %s seconds ---" % (time.time() - start_time))
        

    #save
    final_df=pd.DataFrame(data,columns=["n_cliente","nearest neighbor - <RowID>","distance","nearest neighbor - index"])
    final_df.to_csv(path_write)
    print("--- Final %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()