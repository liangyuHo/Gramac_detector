import numpy as np
import pandas as pd
import pickle



def main():
    label_dict = {0:'BenignWare', 1:'Malware'}
    feature = np.load('../feature/feature.npy')
    model_path = "./model_save/detection_model"
    model = pd.read_pickle(model_path)
    result = model.predict([feature])
    result = np.argmax(result)
    print("result :",label_dict[result])

if __name__=='__main__':
    
    main()


