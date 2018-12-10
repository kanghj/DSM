import argparse

import dsm


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='input_traces',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--work_dir', type=str, default='work_dir',
                       help='directory to output a FSA')

    parser.add_argument('--max_cpu', type=int, default=4,
                       help='Maximum number of processors for parallel processing')

    ##### RNNLM Learning parameters ####

    parser.add_argument('--rnn_size', type=int, default=32,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=25,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=10000,
                       help='save frequency')
    # parser.add_argument('--gpu_mem', type=float, default=0.666,
    #                     help='% of gpu memory to be allocated to this process. Default is 66.6%')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)

    #### parameters for updating an existing automaton ####

    parser.add_argument('--old_fsm', type=str, default=None,
                       help='Path to a file containing the existing automaton')
    parser.add_argument('--additional_trace', type=str, default=None,
                       help='Path to a folder containing input.txt that has additional traces for updating the existing automaton')

    #### clustering parameters ####

    parser.add_argument('--max_cluster', type=int, default=20,
                        help='Max. number of clusters setting')
    parser.add_argument('--min_cluster', type=int, default=2,
                        help='Min. number of clusters setting')
    parser.add_argument('--seed', type=int, default=9999,
                        help='Initialized seed')
    parser.add_argument('--dbscan_eps', type=float, default=0.1,
                        help='DBSCAN\'s eps parameter (for future release)')
    parser.add_argument('--dfa', type=int, default=1,
                        help='Create DFA')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    input_option = read_args()

    dsm.learn_model(input_option.data_dir + '/input.txt', input_option.save_dir, input_option.work_dir, input_option)
