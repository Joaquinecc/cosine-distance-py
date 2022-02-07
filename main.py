import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix, vstack,csr_matrix
from scipy.sparse import lil_matrix
import numpy as np
import time
from multiprocessing import  Pool,cpu_count
import json
 

start_time = time.time()
BYTES_TO_MB_DIV = 0.000001

'''

Global Variable

'''

# Opening JSON file setting
f = open('settings-params.json') 
#Read setting params
settings_var = json.load(f)
path=settings_var['read_path']
path_write=settings_var['path_write']
chunksize=settings_var['chunksize']
topNProduct=settings_var['topNProduct']
topNSimilarity=settings_var['topNSimilarity']

def get_csr_memory_usage(X_csr):
    mem = (X_csr.data.nbytes + X_csr.indptr.nbytes + X_csr.indices.nbytes) * BYTES_TO_MB_DIV
    print("Memory usage is " + str(mem) + " MB")

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
def ranking_similarity_client(cx,index,N=10):
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


def rank_sku(args):
    '''
    Chose the N product with the highest weigh

    Params
    df_ratings: Panda dataframe : containig user ratings
    rows: Panda dataframe : containing neighbor cosine similarity
    N: unit:  Number of product to recomend

    return: 
    Panda Dataframe, with columns 0 [n_client,producto,weigth,rank]
    
    '''
    df_ratings,rows,n_client,N = args
    print(n_client)

    #Neighbor ratings products 
    df_rating_neighbor= df_ratings.loc[rows["nearest neighbor - <RowID>"]].drop(["Cluster_Cuantitativo"], axis=1)
    #drop producto columns fill with 0 (product that where not consumed)
    df_rating_neighbor=df_rating_neighbor.loc[:, (df_rating_neighbor != 0).any(axis=0)]


    #Drop products n_client already consume
    #df_rating_n_client= df_rating_neighbor.loc[[n_client]]
    df_rating_n_client= df_ratings.loc[[n_client]].drop(["Cluster_Cuantitativo"], axis=1)
    df_rating_n_client=df_rating_n_client.loc[:, (df_rating_n_client != 0).any(axis=0)]
    product_already_consumed=df_rating_n_client.columns.tolist()
    df_rating_neighbor=df_rating_neighbor.drop(product_already_consumed, axis=1)


    #Calc ratingXdistance
    df_rating_neighbor["distance"]=rows["distance"].to_numpy().astype(float)
    for cols in df_rating_neighbor:
        if cols != "distance":
            df_rating_neighbor[cols]=df_rating_neighbor[cols]*df_rating_neighbor["distance"]
    
    #Calc weights
    df_temp=df_rating_neighbor.sum()
    if (df_temp["distance"] == 0):
        df_temp=pd.DataFrame(np.full(df_temp.shape, 0.1),index=df_temp.index,columns=["weigth"]).drop(["distance"],axis=0)
    else:
        df_temp=df_temp.div(df_temp["distance"], axis=0, fill_value=0)
        df_temp=df_temp.drop(["distance"],axis=0)
        df_temp=pd.DataFrame((df_temp),columns=["weigth"])
    
    #current client and product"
    df_temp["producto"]=df_temp.index
    df_temp["n_client"]=[n_client]*len(df_temp)
    #rank by weight
    df_temp=df_temp.sort_values(by=['weigth'],ascending=False)
    df_temp["rank"]=np.arange(1,df_temp.shape[0]+1)
    df_temp=df_temp[df_temp["rank"]<=N]    
    return df_temp

def main():
    print("Start..")
    print("--- %s seconds ---" % (time.time() - start_time))

    #Get DATA by chunks and transform to spare
    chunks=pd.read_csv(path,converters={'n_cliente' : str, 'Cluster_Cuantitativo':str}, chunksize=5000)
    sp_data = []
    product_list=[]
    clusters=np.array([])
    clients_row=np.array([])
    for chunk in chunks:
        data=csr_matrix(chunk.drop(['Cluster_Cuantitativo',"n_cliente"], axis=1))
        sp_data.append(data)
        if(len(product_list)==0):
            product_list=chunk.columns[( chunk.columns != "n_cliente" ) & (chunk.columns != "Cluster_Cuantitativo")]
            product_list=[ sku.replace("COD_","") for sku in product_list]
        clients_row=np.concatenate((clients_row,chunk['n_cliente'].to_numpy()))
        clusters=np.concatenate((clusters,chunk['Cluster_Cuantitativo'].to_numpy()))
    sp_data = vstack(sp_data)
    get_csr_memory_usage(sp_data)
    print("--- Reading %s seconds ---" % (time.time() - start_time))
    #transform to spare panda, easier to filter by cluster
    df_ratings=pd.DataFrame.sparse.from_spmatrix(sp_data,index=clients_row,columns=product_list)
    print_memory_usage_of_data_frame(df_ratings)
    df_ratings['Cluster_Cuantitativo'] = clusters


    #Calc distance btw clients and ranking by similarity
    cluster_group = df_ratings.groupby("Cluster_Cuantitativo")
    data=[]
    for cluster, rows in cluster_group:
        indexs=rows.index
        #Calc Cosine similarity
        sp_cl=csr_matrix(rows.drop(['Cluster_Cuantitativo'], axis=1))
        sp_cl=cosine_similarity(sp_cl,dense_output=False)
        sp_cl=coo_matrix(sp_cl)
        #calc rank data
        sp_cl=ranking_similarity_client(sp_cl,indexs,N=topNSimilarity)
        if(len(data) == 0):
            data=sp_cl.copy()
        else:
            data = np.concatenate((data, sp_cl))
    print("--- Ranking similarity by  client %s seconds ---" % (time.time() - start_time))
        

        
    df_rank=pd.DataFrame(data,columns=["n_client","nearest neighbor - <RowID>","distance","nearest neighbor - index"])
    clients_group = df_rank.groupby("n_client")
    df_ranking_sku=pd.DataFrame([],columns=["weigth","producto","n_client","rank"])

    with Pool(cpu_count()) as pool:
        df_ranking_sku=df_ranking_sku.append(pool.map( rank_sku, [(df_ratings,rows,n_client,topNProduct) for n_client, rows in clients_group]))


    #save
    df_ranking_sku.to_csv(path_write)
    print("--- Final %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()