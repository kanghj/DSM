import fsa_construction.k_ptails as feature_extractor
import fsa_construction.clustering_pro as clustering_processing
import fsa_construction.Standard_Automata
import fsa_construction.input_processing as input_processing

import argparse
from collections import defaultdict


debug = False


def main():
    # Run this command from the data directory of the class (e.g. data/ZipOutputStream/) to test:
    # python3 ../../DSM_updater.py new_traces/traces.txt
    args = options()
    if args.debug is True:
        print("In debug mode")
        global debug
        debug = True

    # reusing code from DSM

    # input_processing.select_traces(args.traces , args.traces + '_selected')

    # writes to args.features4clustering_dir as a side effect
    # reads from args.cluster_trace_file
    feature_extractor.feature_engineering(OptionsSubset(args))

    method_list = method_names(args.methods)

    # collect X (feature vectors for each node)
    X, generated_traces, _, method_list, _ = \
        clustering_processing.parse_sampled_traces(args.feature_dir, 'd', method_list=method_list)

    node_coords, cluster_nodes = clustering_processing.read_cluster_contents("work_dir" + "/FINAL_resultant_cluster.gz")

    # write to file for easy debugging
    clustering_processing.write_X_to_file(X, method_list, generated_traces, "work_dir" + '/new_traces_X.txt')

    # cluster representative node is the node nearest to the centroid of each cluster
    cluster_representative = cluster_representative_node(args.cluster_distances, node_coords)

    # find max dist of furthest node from each cluster's representative node
    max_dist = max_distance_from_cluster_representative_node(cluster_representative, cluster_nodes, node_coords)

    element_id_to_cluster = find_nearest_cluster_for_each_node(X, cluster_representative, max_dist)

    # load old FSM
    automata = clustering_processing.StandardAutomata.deserialize(args.fsa)

    # update old FSM
    fsm, log_fsm = clustering_processing.update_fsm(automata, element_id_to_cluster, generated_traces)
    mindfa = fsa_construction.Standard_Automata.minimize_dfa(fsm.nfa2dfa())

    # write updated FSM
    with open("work_dir" + '/new_fsm.txt', 'w') as fff:
        fff.write(fsm.to_string())
    with open("work_dir" + '/new_dfa.txt', 'w') as fff:
        fff.write(mindfa.to_string())
    clustering_processing.drawing_dot(fsm, "work_dir" + '/new_fsm')
    clustering_processing.drawing_dot(mindfa, "work_dir" + '/new_min_dfa')
    print("===done===")


def options():

    parser = argparse.ArgumentParser(description='Update an existing model with new traces')
    parser.add_argument('traces', metavar='t', type=str,
                        help='path to file containing new traces')
    parser.add_argument('--fsa', type=str, default="work_dir/FINAL_serialized_fsa.json",
                        help='path to existing fsa')
    parser.add_argument('--methods', type=str, default="work_dir/method_list.txt",
                        help='path to file containing list of methods')
    parser.add_argument('--cluster_centroids', type=str, default="work_dir/FINAL_centroids.txt",
                        help='path to file containing the cluster centroids')
    parser.add_argument('--cluster_contents', type=str, default="work_dir/FINAL_resultant_cluster.gz",
                        help='path to file containing the cluster contents/nodes')
    parser.add_argument('--cluster_distances', type=str, default="work_dir/FINAL_cluster_element_distances.txt",
                        help='path to file containing cluster element distances from centroid')
    parser.add_argument('--rnn', type=str, default="saved_model/FINAL_cluster_element_distances.gz",
                        help='path to file containing cluster element distances from centroid')
    parser.add_argument('--max_cpu', type=int, default=4,
                        help='Maximum number of processors for parallel processing')
    parser.add_argument('--save_dir', type=str, default='saved_model',
                        help='directory to store checkpointed models')
    parser.add_argument('--feature_dir', type=str, default='new_features4clustering',
                        help='directory to save features extracted for each node')
    parser.add_argument('--debug', type=bool, default=False,
                        help='debug mode')

    return parser.parse_args()


class OptionsSubset:
    def __init__(self, args):
        # these field names are based on DSM.py's Option.
        self.features4clustering_dir = args.feature_dir
        self.cluster_trace_file = args.traces #+ '_selected'
        self.args = args


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


if __name__ == '__main__':
    main()
