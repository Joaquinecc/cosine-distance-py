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


def get_csr_memory_usage(spare_csr):
    '''
    Calc memory usage of a spare matrix and print it
    
    Params:
    spare_csr: spare scipy matrix

    '''
    mem = (spare_csr.data.nbytes + spare_csr.indptr.nbytes + spare_csr.indices.nbytes) * BYTES_TO_MB_DIV
    print("Memory usage is " + str(mem) + " MB")

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
            temp=sorted(temp, key=lambda d: d['peso']) 
            #only rank the top N element
            [element.update({"rank":index+1}) for index,element in enumerate(temp[0:N]) ]
            for element in temp[0:N]:
                            #Current Client       Neighbor       cosine distance  rank position"
                data.append([index[current_row],element["vecino"],element["peso"],element["rank"]])
            current_row=i
            temp=[]
                                        #cosine distance
        temp.append({"vecino":index[j],"peso":1-v})
    return data


def main():
    print("Start..")
    print("--- %s seconds ---" % (time.time() - start_time))

    #Read csv file in chunks so memory not colapse
    chunks=pd.read_csv(path,index_col="n_cliente",converters={'n_cliente' : str}, chunksize=chunksize)
    sp_data = []
    columns=[]
    index=np.array([])
    #transform data to sparse matrix, to save memory and improve perfomance
    for chunk in chunks:
        data=csr_matrix(chunk.drop(['Cluster_Cuantitativo'], axis=1))
        sp_data.append(data)
        if(len(columns)==0):
            columns=chunk.columns
        index=np.concatenate((index,chunk.index))
    sp_data = vstack(sp_data)
    print("--- Reading %s seconds ---" % (time.time() - start_time))

    #print memory in use
    get_csr_memory_usage(sp_data)
    #calc cosine similarity
    df_cosine=cosine_similarity(sp_data,dense_output=False)
    cx=coo_matrix(df_cosine)
    print("--- cosine_similarity %s seconds ---" % (time.time() - start_time))

    rank_data=calc_rank(cx,index,N=30)
    print("--- calc_rank %s seconds ---" % (time.time() - start_time))

    #save
    df=pd.DataFrame(rank_data,columns=["n_client","vecino","peso","rank"])
    df.to_csv(path_write)
    print("--- Final %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()