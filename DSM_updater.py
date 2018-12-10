import argparse
import dsm

def main():
    # Run this command from the data directory of the class (e.g. data/ZipOutputStream/) to test:
    # python3 ../../DSM_updater.py new_traces/traces.txt
    args = options()
    dsm.update_model(args.traces, args.save_dir, args.work_dir, args.output_dir)


def options():
    parser = argparse.ArgumentParser(description='Update an existing model with new traces')
    parser.add_argument('traces', metavar='t', type=str,
                        help='path to file containing new traces')
    parser.add_argument('--fsa', type=str, default="FINAL_serialized_fsa.json",
                        help='path to existing fsa')
    parser.add_argument('--methods', type=str, default="method_list.txt",
                        help='path to file containing list of methods')
    parser.add_argument('--cluster_centroids', type=str, default="FINAL_centroids.txt",
                        help='path to file containing the cluster centroids')
    parser.add_argument('--cluster_distances', type=str, default="FINAL_cluster_element_distances.txt",
                        help='path to file containing cluster element distances from centroid')
    parser.add_argument('--max_cpu', type=int, default=4,
                        help='Maximum number of processors for parallel processing')
    parser.add_argument('--save_dir', type=str, default='saved_model',
                        help='directory to store checkpointed models')
    parser.add_argument('--feature_dir', type=str, default='new_features4clustering',   # used for extracting features
                        help='directory to save features extracted for each node')
    parser.add_argument('--debug', type=bool, default=False,
                        help='debug mode')
    parser.add_argument('--work_dir', type=str, default='work_dir')
    parser.add_argument('--output_dir', type=str, default='updater_output_dir')

    return parser.parse_args()


if __name__ == '__main__':
    main()
