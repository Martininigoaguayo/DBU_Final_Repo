from libs.prefect_helpers import *
from libs.data_manipulation import *
from libs.feature_generation import *
from libs.dim_reduction import *
from libs.football_plots import *
from libs.clustering import *
from prefect import task, flow

@flow
def ripleys_k_flow(name : str):
    df : pd.DataFrame = task_wrapper(compile_team_tracking_data, use_cache=True)("data", name)
    df = task_wrapper(extract_one_match,use_cache=False)(df,1)

    ripleys_k_vals = task_wrapper(ripley_k_by_indices, use_cache=True)(df, df.iloc[::48].index)
    pca_obj = PCAObject(ripleys_k_vals, n_components=10)
    np_pca = task_wrapper(pca_obj.transform,use_cache=False)(ripleys_k_vals)


    #Plotting PC1 and PC2
    plt.scatter(np_pca[:,0], np_pca[:,1])
    plt.savefig(name + "_ripleysk")
    plt.clf()


    clusterer = KMeansObject(np_pca, 5)
    labels = clusterer.get_labels()

    cluster_0_sample = labels.tolist().index(0)
    cluster_1_sample = labels.tolist().index(1)
    cluster_2_sample = labels.tolist().index(2)
    cluster_3_sample = labels.tolist().index(3)
    cluster_4_sample = labels.tolist().index(4)



    generate_pitches_from_start_indices([cluster_0_sample,cluster_1_sample,cluster_2_sample,cluster_3_sample,cluster_4_sample], df, "ripleysk_cluster_examples",5, 100)


    scatter = plt.scatter(np_pca[:, 0], np_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)

    # Create a legend
    legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend1)


    plt.savefig(name + "_ripleysk_clustered")



ripleys_k_flow("Denmark")