from transformers import pipeline, AutoModel, AutoTokenizer
import pandas as pd
import torch
import numpy as np
import time
import selfies as sf


hpf_path = "/hpf/largeprojects/ccmbio/monikas/packages/"
is_BBBP = False

#For Ames dataset
# parent_folder = "Ames_smi/"
# input_file = parent_folder + "Ames.csv"
# smile_column = "Canonical_Smiles"

#For BBBP dataset
parent_folder = "bbbp_smi/"
input_file = parent_folder + "BBBP.csv"
is_BBBP = True
smile_column = "smiles"

#For ClinTox dataset
# parent_folder = "ClinTox_smi/"
# input_file = parent_folder + "ClinTox.csv"
# smile_column = "smiles"

output_path = parent_folder + "feature_outputs/"

model1 = "msb-roshan/molgpt"
model2 = "gayane/BARTSmiles"
model3 = "zjunlp/MolGen-7b" #Needs SELFIES notation
model4 = "DeepChem/ChemBERTa-77M-MTR"
model5 = "ncfrey/ChemGPT-1.2B"


def get_data():
    if is_BBBP:
        encoding='latin-1'
    else:
        encoding = 'utf-8'
    df = pd.read_csv(input_file, encoding=encoding)
    print(f"Testing: {df}")
    return df


''' ORIGINAL
def get_features(extractor, conversion_to_selfie):
    df = get_data()
    if conversion_to_selfie:
        df[smile_column] = df[smile_column].apply(sf.encoder)
    features = extractor(df[smile_column].tolist(), return_tensors = "pt")
    return features

def feature_extraction(model_path, conversion_to_selfie=False):
    print("Running Model:", model_path)
    outfile = model_path.split('/')[1]
    # extractor = pipeline("feature-extraction", framework="pt", model = model_path, model_kwargs={'cache_dir': hpf_path})
    extractor = pipeline("feature-extraction", framework="pt", model = model_path, model_kwargs={'cache_dir': hpf_path}, tokenize_kwargs ={'return_token_type_ids': False})
    features = get_features(extractor, conversion_to_selfie)
    out_filename = output_path + "features_"+ outfile + ".pt"
    torch.save(features, out_filename)
'''

def remove_long_molecules(df):
    """to handle the index out of range error due to max positional embedding limit of 128 in Model2, long molecules with token length > 128 are removed"""
    tokenizer = AutoTokenizer.from_pretrained(model2)
    counter = 0
    # filtered = df[smile_column]
    # idx_to_remove = []

    mask = df[smile_column].apply(lambda smiles: len(tokenizer(smiles).input_ids) > 128)
    counter = mask.sum()
    filtered = df[smile_column][~mask].reset_index(drop=True)

    # for idx, row in df.iterrows():
    #     elem = row[smile_column]
    #     # print(f"Iterator: {idx} {elem}")
    #     tokens = tokenizer(elem)
    #     token_len = len(tokens.input_ids)
    #     # print(f"Tokenizer len: {token_len}")
    #     if token_len > 128:
    #         # print(f'Molecule:', i)
    #         # print(f'Tokens: {tokens}')
    #         idx_to_remove.append(idx)
    #         counter += 1
    # print(f"Items to remove: {idx_to_remove}")
    # filtered = filtered.drop(idx_to_remove).reset_index(drop=True)
    print(f"Counter: {counter}, {len(df[smile_column])}, \nfraction: {(counter/len(df[smile_column]))*100}" )
    # print(f'Filtered: {filtered}')
    return filtered

def get_features(extractor, is_BARTSmiles, conversion_to_selfie):
    df = get_data()
    # print(f'Data: {df} {len(df)}')
    if conversion_to_selfie:
        df[smile_column] = df[smile_column].apply(sf.encoder)
    # print(f'Before extractor')
    final_molecules = df[smile_column]

    if is_BARTSmiles:
        final_molecules = remove_long_molecules(df)

    print(f"Type: {type(final_molecules)}")
    features = extractor(final_molecules.tolist(), return_tensors = "pt")
    return features

def feature_extraction(model_path, is_BARTSmiles, conversion_to_selfie=False):
    print("Running Model:", model_path)
    outfile = model_path.split('/')[1]
    extractor = pipeline("feature-extraction", framework="pt", model = model_path, model_kwargs={'cache_dir': hpf_path}, tokenize_kwargs ={'return_token_type_ids': False})
    features = get_features(extractor,is_BARTSmiles, conversion_to_selfie)
    out_filename = output_path + "features_"+ outfile + ".pt"
    torch.save(features, out_filename)


start_time = time.time()

# feature_extraction(model1)
feature_extraction(model2, True) #ERROR: unexpected 'token_type_ids'
# features = feature_extraction(model3, True)  #Works for Ames

#Error for BBBP in model 3 above
#selfies.exceptions.EncoderError: input violates the currently-set semantic constraints
        # SMILES: c1(CC(N2[C@H](CN(CC2)C(=O)C)C[N@]2CC[C@H](O)C2)=O)ccc(N(=O)=O)cc1
        # Errors:
        # [N with 5 bond(s) - a max. of 3 bond(s) was specified]
#Error for ClinTox and model 3 above
#selfies.exceptions.SMILESParserError: 
# SMILES: *C(=O)[C@H](CCCCNC(=O)OCCOC)NC(=O)OCCOC
        #         ^
        # Index:  0
        # Reason: unrecognized symbol '*'

# feature_extraction(model4) #ChemBERT
# feature_extraction(model5) #ChemGPT


end_time = time.time()
total_time = end_time-start_time
print("Time:", total_time)

