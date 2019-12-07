please install the following packages to run the code
dgl  ( command for windows "conda install -c dglteam dgl")   or follow the instructions: https://www.dgl.ai/pages/start.html
pytorch  (conda install pytorch torchvision cudatoolkit=10.1 -c pytorch) or follow the instructions from the website
networkx   conda install networkx   or follow the instructions
numpy
pandas

dataset files must be placed in the same directory. files are : edge_list.csv, labels.npy, mask.npy

****************************run the code *******************
python GCN_implementation.py


************************************************************





the code generate the validation loss plot and save the results in csv file. 