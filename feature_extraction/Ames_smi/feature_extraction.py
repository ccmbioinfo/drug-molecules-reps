from transformers import pipeline
import pandas as pd
import torch
import numpy as np
import time
import selfies as sf

#Dummy code for test
'''
checkpoint = "facebook/bart-base"
feature_extractor = pipeline("feature-extraction", framework="pt", model=checkpoint)
text = "Transformers is an awesome library!"

#Reducing along the first dimension to get a 768 dimensional array
features = feature_extractor(text,return_tensors = "pt")[0].numpy().mean(axis=0)

print(features)
'''
np.set_printoptions(threshold=np.inf) #for expanding ellipses in text file

input_path = "./Ames.csv"
output_path = "./feature_outputs/"
hpf_path = "/hpf/largeprojects/ccmbio/monikas/packages/"

def get_data():
    df = pd.read_csv(input_path)
    return df

def get_features(extractor, conversion_to_selfie):
    print("Here1")
    df = get_data() #ASK if we have to concatenate columns? or just use the structure
    # print(df)
    # features = df["Canonical_Smiles"].apply(lambda x: extractor(x, return_tensors = "pt")[0].numpy().mean(axis=0))
    if conversion_to_selfie:
        df["Canonical_Smiles"] = df["Canonical_Smiles"].apply(sf.encoder)
    # print(df["Canonical_Smiles"])
    features = extractor(df["Canonical_Smiles"].tolist(), return_tensors = "pt")
    # print(features.get_shape())
    # print(features[0])
    # print(features)
    return features

def feature_extraction(model_path, conversion_to_selfie=False):
    print("Running Model:", model_path)
    outfile = model_path.split('/')[1]
    # print("Here")
    extractor = pipeline("feature-extraction", framework="pt", model = model_path, model_kwargs={'cache_dir': hpf_path})
    features = get_features(extractor, conversion_to_selfie)
    # out_filename = output_path + "features_"+ outfile + ".txt"
    out_filename = output_path + "features_"+ outfile + ".pt"

    torch.save(features, out_filename)

    # print(f"Shape: {type(features)}")
    '''ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (6512, 1) + inhomogeneous part.
    numpy_arrays = [tensor.numpy() for tensor in features]
    np.save(out_filename, numpy_arrays)
    #GPT says it is due to all tensors not being of the same size, so first make sure their shape is same. Look into THIS
    '''
    
    ''' #Too painful to process numpy while reading and writing, doing torch format for now, won't be readable
    with open(out_filename, "w") as file:
        for feature in features:
            # print("Feature:", feature[0][0])
            file.write(np.array2string(feature.numpy(), separator=",") + "\n")
            # file.write(str(feature.numpy()) + "\n") #cannot customize the string output
    #'''


#For Ames dataset
model1 = "msb-roshan/molgpt"
model2 = "gayane/BARTSmiles"
model3 = "zjunlp/MolGen-7b" #Needs SELFIES notation
model4 = "DeepChem/ChemBERTa-77M-MTR"
model5 = "ncfrey/ChemGPT-1.2B"
start_time = time.time()
# feature_extraction(model1)
# feature_extraction(model2) #ERROR: unexpected 'token_type_ids'
# features = feature_extraction(model3, True)  #Errors out in mapping model file in pipeline statement I think
# feature_extraction(model4)
feature_extraction(model5) #Takes too long, 18m!

end_time = time.time()
total_time = end_time-start_time
print("Time:", total_time)

