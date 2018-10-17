import fsa_construction.k_ptails as feature_extractor
import fsa_construction.clustering_pro as clustering_processing
import fsa_construction.Standard_Automata

import argparse


def main():
    # Run this command to test: python3 ../../DSM_updater.py new_traces/traces.txt
    args = options()

    # reusing code from DSM
    # writes to args.features4clustering_dir as a side effect
    # reads from args.cluster_trace_file
    feature_extractor.feature_engineering(OptionsSubset(args))

    method_list = method_names(args.methods)

    # collect X (feature vectors for each node)
    X, generated_traces, additional_val_traces, method_list, possible_ending_words = \
        clustering_processing.parse_sampled_traces(args.feature_dir, 'd', method_list=method_list)

    # first read the cluster coords and max dist from each cluster
    cluster_centroids = read_cluster_centroids(args.cluster_centroids)
    max_dist = max_distance_from_cluster_centroids(args.cluster_distances)

    # write to file for easy debugging
    clustering_processing.write_X_to_file(X, method_list, generated_traces, "work_dir" + '/new_traces_X.txt')
    # for each node, select the nearest cluster
    element_id_to_cluster = find_nearest_cluster_for_each_node(X, cluster_centroids, max_dist)

    # load old FSM
    automata = clustering_processing.StandardAutomata([], [], []).deserialize(args.fsa)

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

    return parser.parse_args()


class OptionsSubset:
    def __init__(self, args):
        self.features4clustering_dir = args.feature_dir
        self.cluster_trace_file = args.traces
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


def find_nearest_cluster_for_each_node(X, cluster_centroids, max_dist):
    element_to_cluster = {}
    epsilon = 0.001
    for i, x in enumerate(X):
        print('i: ' + str(i))

        # find nearest cluster for each X
        eliglible_clusters_and_dist = [(cluster_id, clustering_processing.compute_distance(coords, x)) for
                                       cluster_id, coords in cluster_centroids.items() if
                                       clustering_processing.compute_distance(coords, x) < max_dist[
                                           cluster_id] + epsilon]
        if len(eliglible_clusters_and_dist) == 0:
            print("no eligible clusters")
            print(x)
            for cluster_id, coords in cluster_centroids.items():
                print(cluster_id + ": " + str(clustering_processing.compute_distance(coords, x)))
        elif len(eliglible_clusters_and_dist) > 2:
            print("OMG, more than 2")
            # pick cluster with smallest_dist
            selected_cluster = min(eliglible_clusters_and_dist, key=lambda x: x[1])
        else:
            # 1 cluster
            selected_cluster = eliglible_clusters_and_dist[0]
            print('single cluster')
            print(eliglible_clusters_and_dist)
        element_to_cluster[str(i)] = selected_cluster[0]
    return element_to_cluster


def max_distance_from_cluster_centroids(cluster_distances_filepath):
    """
    Reads from the cluster distances file and determine the furthest distance
    :param cluster_distances_filepath:
    :return: max_dist
    """
    max_dist = {}
    with open(cluster_distances_filepath, 'r') as cluster_distance_file:
        for line in cluster_distance_file:
            cluster_id = line.split()[0]
            distance = float(line.split()[-1])
            if cluster_id not in max_dist or max_dist[cluster_id] < distance:
                max_dist[cluster_id] = distance
    return max_dist


if __name__ == '__main__':
    main()
