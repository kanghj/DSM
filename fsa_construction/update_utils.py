from collections import defaultdict

import fsa_construction.k_ptails as feature_extractor
import fsa_construction.clustering_pro as clustering_processing
import fsa_construction.Standard_Automata


debug = False


def method_names(methods):
    # note that method_list needs to be sorted (in the same order as the original DSM).
    # We assume here that it's sorted in the file.
    with open(methods, 'r') as method_file:
        method_list = [w.strip() for w in method_file]
    return method_list


def read_cluster_centroids(cluster_centroids_filepath):
    cluster_centroids = {}
    with open(cluster_centroids_filepath, 'r') as cluster_centroid_file:
        for line in cluster_centroid_file:
            cluster_id = line.split()[0]
            coords = [float(val) for val in line.split()[1:]]

            cluster_centroids[cluster_id] = coords
    return cluster_centroids


def find_nearest_cluster_for_each_node(X, cluster_center, max_dist):
    element_to_cluster = {}
    epsilon = 0.001
    selected_cluster = None
    for i, x in enumerate(X):
        if debug:
            print('i: ' + str(i))
        # find nearest cluster for each X
        eliglible_clusters_and_dist = [(cluster_id, clustering_processing.compute_distance(coords, x)) for
                                       cluster_id, coords in cluster_center.items() if
                                       clustering_processing.compute_distance(coords, x) < max_dist[
                                           cluster_id] + epsilon]
        if len(eliglible_clusters_and_dist) == 0:
            if debug:
                print("no eligible clusters")
                print(x)
                for cluster_id, coords in cluster_center.items():
                    print(cluster_id + ": " + str(clustering_processing.compute_distance(coords, x)))
        elif len(eliglible_clusters_and_dist) > 2:
            # print("more than 2")
            # pick cluster with smallest_dist
            selected_cluster = min(eliglible_clusters_and_dist, key=lambda x: x[1])
        else:
            # 1 cluster
            selected_cluster = eliglible_clusters_and_dist[0]
            # print('single cluster')
            # print(eliglible_clusters_and_dist)
        if selected_cluster is not None and len(selected_cluster) > 0:
            element_to_cluster[str(i)] = selected_cluster[0]
        else:
            element_to_cluster[str(i)] = -1

    return element_to_cluster


def max_distance_from_cluster_representative_node(cluster_representative, cluster_nodes, node_coords):
    """
    Determines the distance of the furthest node from the cluster's representative node.
    :param cluster_representative: map of cluster_id -> coords of representative node
    :param cluster_nodes: map of cluster_id -> list of node ids
    :param node_coords: map of node id -> coords
    :return:
    """
    max_dist = defaultdict(lambda : 0)
    for cluster, nodes in cluster_nodes.items():
        representative_node_coords = cluster_representative[cluster]
        for node in nodes:
            dist = clustering_processing.compute_distance(node_coords[node], representative_node_coords)
            if dist > max_dist[cluster]:
                max_dist[cluster] = dist
    return max_dist


def cluster_representative_node(cluster_distances_filepath, node_coords):
    min_dist = {}
    cluster_representative = {}
    with open(cluster_distances_filepath, 'r') as cluster_distance_file:
        for line in cluster_distance_file:
            splitted = line.split()
            cluster_id = splitted[0]
            node_id = splitted[1]
            distance = float(splitted[-1])
            if cluster_id not in min_dist or min_dist[cluster_id] > distance:
                min_dist[cluster_id] = distance
                ID = node_id.lstrip("ID:")
                cluster_representative[cluster_id] = node_coords[ID]
    return cluster_representative

