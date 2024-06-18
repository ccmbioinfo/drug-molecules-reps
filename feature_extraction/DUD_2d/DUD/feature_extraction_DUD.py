from transformers import pipeline
import pandas as pd
import torch
import time
import selfies as sf


hpf_path = "/hpf/largeprojects/ccmbio/monikas/packages/"
dataset = 'datafiles/'
output_path = "feature_outputs/"
smile_column = "SMILES"
label_column = 'Labels'
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

model1 = "msb-roshan/molgpt"
model2 = "gayane/BARTSmiles"
model3 = "zjunlp/MolGen-7b" #Needs SELFIES notation
model4 = "DeepChem/ChemBERTa-77M-MTR"
model5 = "ncfrey/ChemGPT-1.2B"

def read_data(input_string):
    #separator for columns in these files is tab
    smiles_actives = pd.read_csv(dataset + "cmp_list_DUD_" + input_string + "_actives.dat", sep='\t')[[smile_column]]
    # print(f'Data: {type(smiles_actives)}')
    smiles_decoys = pd.read_csv(dataset + "cmp_list_DUD_" + input_string + "_decoys.dat", sep='\t')[[smile_column]]
    smiles_actives[label_column] = 'actives'
    smiles_decoys[label_column] = 'decoys'
    new_df = pd.concat([smiles_actives, smiles_decoys], axis=0, ignore_index=True)
    print(f'New: {new_df}')
    return new_df

# '''
def get_features(extractor, input_str, conversion_to_selfie):
    df = read_data(input_str)
    if conversion_to_selfie:
        df[smile_column] = df[smile_column].apply(sf.encoder) #No invalid selfies in DUD, so no handler here
    features = extractor(df[smile_column].tolist(), return_tensors = "pt")
    return features, df[label_column]

def feature_extraction(model_path, input_str, conversion_to_selfie=False):
    start_time = time.time()
    print("Running Model:", model_path, "for:", input_str, "at:", time.time())
    outfile = model_path.split('/')[1]
    extractor = pipeline("feature-extraction", framework="pt", model = model_path, model_kwargs={'cache_dir': hpf_path}, tokenize_kwargs ={'return_token_type_ids': False})
    features, labels = get_features(extractor, input_str, conversion_to_selfie)
    out_filename = output_path + "features_labels_"+ input_str + "_" + outfile + ".pt"
    tensor_wth_labels = {'tensors': features, 'labels': labels}
    torch.save(tensor_wth_labels, out_filename)
    end_time = time.time()
    print("Saved file at:", end_time - start_time)
    # torch.save(features, out_filename)
# '''

start_time = time.time()

# feature_extraction(model1, input_string) #MolGPT
# feature_extraction(model2, target_list[1]) #ERROR: unexpected 'token_type_ids'
# features = feature_extraction(model3, True)  #Errors out in mapping model file in pipeline statement I think
# feature_extraction(model4, input_string) #ChemBERT
# feature_extraction(model5, input_string) #ChemGPT

#Loop for generating all .pt files
for target in target_list.values():
    feature_extraction(model1, target) #MolGPT
    feature_extraction(model2, target) #BartSmiles
    feature_extraction(model3, target, True) #MolGen
    feature_extraction(model4, target) #ChemBERT
    feature_extraction(model5, target) #ChemGPT

end_time = time.time()
total_time = end_time-start_time
print("Time:", total_time)