import os,sys
import multiprocessing
import shutil

import dsm.train as RNNLM_training
import fsa_construction.input_processing as input_sampler
import fsa_construction.estimate_accuracy as model_selection

import fsa_construction.k_ptails as feature_extractor
import fsa_construction.clustering_pro as clustering_processing
import fsa_construction.Standard_Automata
import fsa_construction.update_utils as update_utils

# import fsa_construction.input_processing as input_processing


class Option:
    def __init__(self, data_dir, rnn_dir, work_dir):
        self.update_mode = False

        ####################################################################################

        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        self.work_dir = work_dir
        self.raw_input_trace_file = data_dir
        self.save_dir = rnn_dir

        #######
        if not os.path.isfile(self.raw_input_trace_file):
            print("Cannot find input execution traces stored in", self.raw_input_trace_file)
            sys.exit(-1)
        # self.preprocessed_trace_file = self.args.data_dir+'/input.txt'
        self.cluster_trace_file = data_dir + '/cluster_traces.txt'
        self.clustering_space_dir = work_dir + '/clustering_space'
        self.features4clustering_dir = work_dir + '/features4clustering'
        ####################################################################################

        self.generated_traces_folder = self.features4clustering_dir
        self.validation_traces_folder = self.raw_input_trace_file
        self.output_folder = self.clustering_space_dir
        self.min_cluster = 2
        self.max_cluster = 20
        self.max_cpu = 4
        self.seed = 9999
        self.dfa = 1
        self.dbscan_eps = 0.1  # apparently this is for future releases?


def run_dsm(input_option, args):
    ######## preprocessing traces & trace sampling ###########
    input_sampler.select_traces(input_option.raw_input_trace_file, input_option.cluster_trace_file)
    ######## train RNNLM model ########
    if not os.path.isdir(input_option.save_dir):

        os.makedirs(input_option.save_dir)

        p = multiprocessing.Process(target=RNNLM_training.train, args=(args,))
        p.start()
        p.join()
        # RNNLM_training.train(input_option.args)
    ######## feature extraction ########
    feature_extractor.feature_engineering(input_option)
    ######## clustering ########
    clustering_processing.clustering_step(input_option)
    ######## model selection #######
    final_file = model_selection.selecting_model(input_option)
    print("Done! Final FSM is stored in", final_file)


def learn_model(input_path, rnn_model_dir, output_dir, args):
    """
    Constructs a new FSA and writes it into output_dir/serialized_fsa.json.
    Writes intermediate outputs such as diagrams of the FSA in output_dir.

    :param input_path: path to file containing input traces
    :param rnn_model_dir: path to directory to store RNN model.
    :param output_dir: path to directory to store final results and other intermediate output.
    :param args:        args for neural network.
                data_dir (str): directory containing training data, should be the same directory that input_path is in.
                rnn_size (int): size of RNN hidden state. Defaults to 32.
                num_layers (int): number of layers in the RNN. Defaults to 2.
                model (str): rnn, gru, or lstm. Defaults to lstm.
                batch_size (int): Minibatch size. Defaults to 10.
                seq_length (int): RNN sequence length. Defaults to 25.
                num_epochs (int): number of epochs. Defaults to 10.
                grad_clip (float): clip gradients at this value. Defaults to 5.
                learning_rate (float): Defaults to 0.002.
                decay_rate (float): decay rate for rmsprop. Defaults to 0.97.
    """

    run_dsm(Option(input_path, rnn_model_dir, output_dir), args)


def accept_traces(traces, fsa_path):
    """
    Given a list of execution traces, returns a list of booleans.
    For each trace in the list, True is returned if the trace is accepted by the FSA, otherwise False.
    :param traces: a list of execution traces
    :param fsa_path: path to FSA built using learn_model
    :return: a list of booleans indicating whether each trace is accepted or rejected
    """
    automata = clustering_processing.StandardAutomata.deserialize(fsa_path)
    fsm_adjlst = automata.create_adjacent_list()

    ret_val = []
    for trace in traces:
        flag, _ = automata.is_accepting_one_trace(trace, fsm_adjlst)
        ret_val.append(flag)
    return ret_val


class OptionsSubset:
    def __init__(self, args):
        # these field names are based on DSM.py's Option.
        self.features4clustering_dir = args.feature_dir
        self.cluster_trace_file = args.traces #+ '_selected'
        self.args = args


class UpdateOptions:
    def __init__(self, traces, fsa, method_path, rnn_dir, feature_dir, cluster_distance_file, prev_work_dir, output_dir):
        self.traces = traces
        self.fsa = fsa
        self.methods = method_path
        self.max_cpu = 4
        self.save_dir = rnn_dir
        self.feature_dir = feature_dir
        self.debug = False
        self.work_dir = prev_work_dir
        self.output_dir = output_dir
        self.cluster_distances = cluster_distance_file


def run_dsm_update(options):

    if options.debug is True:
        print("In debug mode")
        global debug
        debug = True

    # reusing code from DSM

    # input_processing.select_traces(args.traces , args.traces + '_selected')

    # writes to args.features4clustering_dir as a side effect
    # reads from args.cluster_trace_file
    feature_extractor.feature_engineering(OptionsSubset(options))

    method_list = update_utils.method_names(options.work_dir + '/' + options.methods)

    # collect X (feature vectors for each node)
    X, generated_traces, _, method_list, _ = \
        clustering_processing.parse_sampled_traces(options.feature_dir, 'd', method_list=method_list)

    node_coords, cluster_nodes = clustering_processing.read_cluster_contents(
        options.work_dir + "/FINAL_resultant_cluster.gz")

    # write to file for easy debugging
    clustering_processing.write_X_to_file(X, method_list, generated_traces, options.work_dir + '/new_traces_X.txt')

    # cluster representative node is the node nearest to the centroid of each cluster
    cluster_representative = update_utils.cluster_representative_node(options.work_dir + '/' + options.cluster_distances,
                                                                      node_coords)

    # find max dist of furthest node from each cluster's representative node
    max_dist = update_utils.max_distance_from_cluster_representative_node(cluster_representative, cluster_nodes,
                                                                          node_coords)

    element_id_to_cluster = update_utils.find_nearest_cluster_for_each_node(X, cluster_representative, max_dist)

    # load old FSM
    automata = clustering_processing.StandardAutomata.deserialize(options.work_dir + '/' + options.fsa)

    # update old FSM
    fsm, log_fsm = clustering_processing.update_fsm(automata, element_id_to_cluster, generated_traces)
    mindfa = fsa_construction.Standard_Automata.minimize_dfa(fsm.nfa2dfa())

    # write updated FSM
    with open(options.output_dir + '/new_fsm.txt', 'w') as fff:
        fff.write(fsm.to_string())
    with open(options.output_dir + '/new_dfa.txt', 'w') as fff:
        fff.write(mindfa.to_string())
    clustering_processing.drawing_dot(fsm, options.output_dir + '/new_fsm')
    clustering_processing.drawing_dot(mindfa, options.output_dir + '/new_min_dfa')

    try:
        fsm.serialize(options.output_dir + "/FINAL_serialized_fsa.json")
    except Exception as e:
        print("Serialization problem:")
        print(e)

    # work for preparing directory for future updates

    # update cluster information

    shutil.copyfile(options.work_dir + '/FINAL_resultant_cluster.gz',
                    options.output_dir + '/FINAL_resultant_cluster.gz')
    shutil.copyfile(options.work_dir + '/FINAL_cluster_element_distances.txt',
                    options.output_dir + '/FINAL_cluster_element_distances.txt')
    # copy method list to output directory
    with open(options.output_dir + '/' + options.methods, 'w+') as method_file:
        for method in method_list:
            method_file.write(method)
            method_file.write('\n')
    print("===done===")


def update_model(input_path, rnn_model_dir, old_fsa_output_dir, output_dir):
    """
    Updates an existing FSA with new traces.
    :param input_path:          path to file containing new traces
    :param rnn_model_dir:       directory containing rnn model
    :param old_fsa_output_dir:  old output directory containing the previous fsa model and related output
    :param output_dir:          output working directory
    """
    options = UpdateOptions(input_path, fsa='FINAL_serialized_fsa.json', method_path='method_list.txt',
                            rnn_dir=rnn_model_dir, feature_dir='new_features4clustering',
                            cluster_distance_file="FINAL_cluster_element_distances.txt",
                            prev_work_dir=old_fsa_output_dir, output_dir=output_dir)

    run_dsm_update(options)
