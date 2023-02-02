import pandas as pd
from stucco import cfg

orig_path = "/home/zhsh/robot_ws/src/stucco/data/poking_real_drill.pkl"
override_path = "/home/zhsh/Downloads/poking_real_drill.pkl"
new_path = "/home/zhsh/robot_ws/src/stucco/data/poking_real_drill_merged.pkl"

d1 = pd.read_pickle(orig_path)
d2 = pd.read_pickle(override_path)

key_columns = ("method", "name", "seed", "poke", "level", "batch")

dcombined = pd.concat([d1, d2])
# clean up the database by removing duplicates (keeping latest results)
df = dcombined.drop_duplicates(subset=key_columns, keep='last')

df.to_pickle(new_path)
