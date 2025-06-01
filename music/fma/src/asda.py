import numpy as np
import pandas as pd

df = pd.read_csv("C:\\Users\\OZBERK\\Desktop\\traffic-music\\music\\fma\\matched_pruned_for_training_with_mels.csv")

# Pick a few track_ids at random:
for tid in [2, 10, 140]:
    mel = np.load(f"C:/Users/OZBERK/Desktop/traffic-music/music/fma/mels/{tid}.npy")
    print(tid, "min, max, mean:", mel.min(), mel.max(), mel.mean(), "any NaN?", np.isnan(mel).any())

