# DSM (Deep Specification Miner) [[pdf](https://github.com/lebuitienduy/DSM/blob/master/paper/DSM.pdf)] [[slides](https://github.com/lebuitienduy/DSM/blob/master/paper/DSM-ISSTA.pptx)]
## Installation
### Linux

- Download and install Anaconda3 4.2 from https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
- After installation, include path of "bin" folder of the above Anaconda to PATH variable
- Install Tensorflow 0.12 for the installed Anaconda version using the command: 
```
conda install -c jjhelmus tensorflow=0.12.0
```
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
- Run `python DSM_updater.py`. From one of the data directories (e.g. data/ZipOutputStream), run `python3 ../../DSM_updater.py new_traces/traces.txt` where `data/ZipOutputStream/new_traces/traces` contains new traces in the same format as the original traces.  