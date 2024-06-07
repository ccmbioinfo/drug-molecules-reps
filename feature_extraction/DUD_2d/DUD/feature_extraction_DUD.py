from transformers import pipeline
import pandas as pd
import torch
import numpy as np
import time
import selfies as sf


hpf_path = "/hpf/largeprojects/ccmbio/monikas/packages/"
parent_folder = "DUD/"
output_path = parent_folder + "feature_outputs/"
smile_column = "SMILES"
# parent_folder = "DUD/"
target_list = {1: 'ace', 
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
               21:'vegfr2'
               }
input_string = target_list[6]

model1 = "msb-roshan/molgpt"
model2 = "gayane/BARTSmiles"
model3 = "zjunlp/MolGen-7b" #Needs SELFIES notation
model4 = "DeepChem/ChemBERTa-77M-MTR"
model5 = "ncfrey/ChemGPT-1.2B"

def read_data(input_string):
    #separator for columns in these files is tab
    smiles_actives = pd.read_csv("cmp_list_DUD_" + input_string + "_actives.dat", sep='\t')[[smile_column]]
    print(f'Data: {type(smiles_actives)}')
    smiles_decoys = pd.read_csv("cmp_list_DUD_" + input_string + "_decoys.dat", sep='\t')[[smile_column]]
    new_df = pd.concat([smiles_actives, smiles_decoys], axis=0, ignore_index=True) #Haven't run with ignore_index yet
    # print(f'New: {new_df}')
    return new_df

# read_data()

def get_features(extractor, input_str, conversion_to_selfie):
    df = read_data(input_str)
    if conversion_to_selfie:
        df[smile_column] = df[smile_column].apply(sf.encoder)
    features = extractor(df[smile_column].tolist(), return_tensors = "pt")
    return features

def feature_extraction(model_path, input_str, conversion_to_selfie=False):
    print("Running Model:", model_path)
    outfile = model_path.split('/')[1]
    extractor = pipeline("feature-extraction", framework="pt", model = model_path, model_kwargs={'cache_dir': hpf_path})
    features = get_features(extractor, input_str, conversion_to_selfie)
    out_filename = output_path + "features_"+ input_str + "_" + outfile + ".pt"
    torch.save(features, out_filename)


start_time = time.time()

# feature_extraction(model1, input_string) #MolGPT
# feature_extraction(model2) #ERROR: unexpected 'token_type_ids'
# features = feature_extraction(model3, True)  #Errors out in mapping model file in pipeline statement I think
# feature_extraction(model4, input_string) #ChemBERT
# feature_extraction(model5, input_string) #ChemGPT

#Loop for generating all .pt files
# for i in range(1,22):
#     feature_extraction(model1, target_list[i]) #MolGPT
#     feature_extraction(model4, target_list[i]) #MolGPT
#     feature_extraction(model5, target_list[i]) #MolGPT

end_time = time.time()
total_time = end_time-start_time
print("Time:", total_time)