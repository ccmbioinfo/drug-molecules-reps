import umap
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import pad
import time


#For DUD dataset
target_list = {
                1: 'ace',
               2: 'ache',
               3: 'ar',
               4: 'cdk2',
               5: 'cox2',
               6: 'dhfr',
               7: 'egfr',
               8: 'er_agonist',
               9: 'fgfr1',
               10: 'fxa',
               11: 'gpb',
               12: 'gr',
               13: 'hivrt',
               14: 'inha',
               15: 'na',
               16: 'p38',
               17: 'parp',
               18: 'pdgfrb',
               19: 'sahh',
               20: 'src',
               21: 'vegfr2'
               }
# input_string = target_list[6]
input_path = "feature_outputs/"
output_path = "umap_outputs/"
dataset = 'datafiles/'

model1 = "msb-roshan/molgpt"
model2 = "gayane/BARTSmiles"
model3 = "zjunlp/MolGen-7b" #Needs SELFIES notation, ERRORs out
model4 = "DeepChem/ChemBERTa-77M-MTR" 
model5 = "ncfrey/ChemGPT-1.2B"

def load_data(target, model):
    filepath = input_path + "features_labels_"+ target + "_" + model + ".pt"
    print(f'Filename: {filepath}')
    # features = torch.load(filepath)
    data = torch.load(filepath)
    features = data['tensors']
    labels = data['labels']
    print(f"Pt file read: \n{len(features)}")
    print(f"Labels: \n{len(labels)}")
    return features, labels

def preprocess_tensors(tensor_list):
    flattened_list = [tensor.view(tensor.size(0), -1) for tensor in tensor_list]
    max_features = max(tensor.size(1) for tensor in flattened_list) #TODO: assuming that dim 1 is the dimension which is different
    # Pad tensors to have the same number of features
    padded_tensors = [pad(tensor, (0, max_features - tensor.size(1))) for tensor in flattened_list]
    concatenated_tensors = torch.cat(padded_tensors, dim=0)
    return concatenated_tensors

def umap_reducer(list, target, labels, model):
    tensors = preprocess_tensors(list)
    data = tensors.numpy()
    reducer = umap.UMAP()
    start_time = time.time()
    print(f"Starting umap reduction at: {start_time}")
    embedding = reducer.fit_transform(data) #reduces to 2D data
    df = pd.DataFrame(embedding, columns=['Dim1', 'Dim2'])
    # print(f"DF: {df}")
    df['Labels'] = labels
    print(f"df: {df}")
    df.to_csv(output_path + "embeddings_" + target + "_" + model + ".csv") #No labels in this dataset, CAS numbers are there, that too missing for some rows
    print(f"Reduction and saving took: {time.time() - start_time}")



# model = model1
# model = model4
# model = model5
# outfile = model.split('/')[1]
# tensor_list = load_data(target_list[6], outfile)
# umap_reducer(tensor_list, target_list[6], outfile)

#for reducing dimensions for all files
for target in target_list.values():
    for m in [model1, model2, model3, model4, model5]:
    # for m in [model5]: #for fxa chem gpt
        model = m.split('/')[1]
        print(f'Running: {target} with {model}')
        tensor_list, labels = load_data(target, model)
        umap_reducer(tensor_list, target, labels, model)

