# DSM (Deep Specification Miner) [[pdf](https://github.com/lebuitienduy/DSM/blob/master/paper/DSM.pdf)] [[slides](https://github.com/lebuitienduy/DSM/blob/master/paper/DSM-ISSTA.pptx)]
## Installation
### Linux

- Download and install Anaconda3 4.2 from https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
- After installation, include path of "bin" folder of the above Anaconda to PATH variable
- Install Tensorflow 0.12 for the installed Anaconda version using the command: 
```
conda install -c jjhelmus tensorflow=0.12.0
```
(Note 3 July 2020: the above command doesn't work anymore. But running `pip install tensorflow==0.12.1` does.)

- Install graphviz for Python:
```
python -m install graphviz
```
- Test installation:
  ```
  cd data/StringTokenizer
  bash execute.sh
  ```

## Updating model with new traces
- When there are new traces that become available after the FSM model has already been constructed, it is possible to update the model without retraining on the entire dataset.
- Much faster than retraining with all available data 
- Run `python DSM_updater.py`. From one of the data directories (e.g. data/ZipOutputStream), run `python3 ../../DSM_updater.py new_traces/traces.txt` where `data/ZipOutputStream/new_traces/traces.txt` contains new traces in the same format as the original traces.
- When using DSM as a library, the update_model API can be used for this.

## Using DSM as a library

- To use DSM as a library, run `python setup.py install` to install the DSM package on your machine. 
  Executing `import dsm` will work if the installation is successful.
- The following 3 APIs are provided:

````
learn_model(input_path: str, rnn_model_dir: str, output_dir: str, args)
    
    Constructs a new FSA and writes it into output_dir/serialized_fsa.json.
    Writes intermediate outputs such as diagrams of the FSA in output_dir.

    :param input_path:      path to file containing input traces
    :param rnn_model_dir:   path to directory that will store the RNN model.
    :param output_dir:      path to directory that will store the final results and other intermediate output.
    :param args:            args for training a neural network. The following attributes can be configured.
                data_dir (str):         directory containing training data, should be the same directory that input_path is in.
                rnn_size (int):         size of RNN hidden state. Defaults to 32.
                num_layers (int):       number of layers in the RNN. Defaults to 2.
                model (str):            rnn, gru, or lstm. Defaults to lstm.
                batch_size (int):       Minibatch size. Defaults to 10.
                seq_length (int):       RNN sequence length. Defaults to 25.
                num_epochs (int):       number of epochs. Defaults to 10.
                grad_clip (float):      clip gradients at this value. Defaults to 5.
                learning_rate (float):  Defaults to 0.002.
                decay_rate (float):     decay rate for rmsprop. Defaults to 0.97.
````

````
accept_traces(traces: Iterable[Iterable[str]], fsa_directory: str)

    Given a list of execution traces, returns a list of booleans.
    For each trace in the list, True is returned if the trace is accepted by the FSA, otherwise False.
    
    :param traces:      a list of execution traces. Each trace is a list of strings.
    :param fsa_directory: path to directory containing FSA built using learn_model. This should be the same value as learn_model's output_dir
    :return:            a list of booleans indicating whether each trace is accepted or rejected
````

````
update_model(input_path: str, rnn_model_dir: str, old_fsa_output_dir: str, output_dir: str)

    Updates an existing FSA with new traces.
    
    :param input_path:          path to file containing new traces
    :param rnn_model_dir:       directory containing rnn model
    :param old_fsa_output_dir:  old output directory containing the previous fsa model and related outputs
    :param output_dir:          output directory for updated FSA

````
