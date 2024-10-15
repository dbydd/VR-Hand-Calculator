# %%
import pandas as pd;

# %%
gesture_0 = pd.read_csv(r"resources/pose_index_0.csv",index_col=False)
# gesture_0 = pd.concat([gesture_0,pd.read_csv(r"resources/pose_index_0_2.csv",index_col=False)],ignore_index=True)
gesture_0['gesture'] = "0"

gesture_1 = pd.read_csv(r"resources/pose_index_1.csv",index_col=False)
# gesture_1 = pd.concat([gesture_1,pd.read_csv(r"resources/pose_index_1_2.csv",index_col=False)],ignore_index=True)
gesture_1['gesture'] = "1"

gesture_2 = pd.read_csv(r"resources/pose_index_2.csv",index_col=False)
# gesture_2 = pd.concat([gesture_2,pd.read_csv(r"resources/pose_index_2_2.csv",index_col=False)],ignore_index=True)
gesture_2['gesture'] = "2"

gesture_3 = pd.read_csv(r"resources/pose_index_3.csv",index_col=False)
# gesture_3 = pd.concat([gesture_3,pd.read_csv(r"resources/pose_index_3_2.csv",index_col=False)],ignore_index=True)
gesture_3['gesture'] = "3"

gesture_4 = pd.read_csv(r"resources/pose_index_4.csv",index_col=False)
# gesture_4 = pd.concat([gesture_4,pd.read_csv(r"resources/pose_index_4_2.csv",index_col=False)],ignore_index=True)
gesture_4['gesture'] = "4"

gesture_5 = pd.read_csv(r"resources/pose_index_5.csv",index_col=False)
# gesture_5 = pd.concat([gesture_5,pd.read_csv(r"resources/pose_index_5_2.csv",index_col=False)],ignore_index=True)
gesture_5['gesture'] = "5"

gesture_6 = pd.read_csv(r"resources/pose_index_6.csv",index_col=False)
# gesture_6 = pd.concat([gesture_6,pd.read_csv(r"resources/pose_index_6_2.csv",index_col=False)],ignore_index=True)
gesture_6['gesture'] = "6"

gesture_7 = pd.read_csv(r"resources/pose_index_7.csv",index_col=False)
# gesture_7 = pd.concat([gesture_7,pd.read_csv(r"resources/pose_index_7_2.csv",index_col=False)],ignore_index=True)
gesture_7['gesture'] = "7"

gesture_8 = pd.read_csv(r"resources/pose_index_8.csv",index_col=False)
# gesture_8 = pd.concat([gesture_8,pd.read_csv(r"resources/pose_index_8_2.csv",index_col=False)],ignore_index=True)
gesture_8['gesture'] = "8"

gesture_9 = pd.read_csv(r"resources/pose_index_9.csv",index_col=False)
# gesture_9 = pd.concat([gesture_9,pd.read_csv(r"resources/pose_index_9_2.csv",index_col=False)],ignore_index=True)
gesture_9['gesture'] = "9"

gesture_compute = pd.read_csv(r"resources/pose_index_compute.csv",index_col=False)
gesture_compute['gesture'] = "compute"
gesture_delete = pd.read_csv(r"resources/pose_index_delete.csv",index_col=False)
gesture_delete['gesture'] = "delete"
gesture_div = pd.read_csv(r"resources/pose_index_div.csv",index_col=False)
gesture_div['gesture'] = "div"
gesture_minus = pd.read_csv(r"resources/pose_index_minus.csv",index_col=False)
gesture_minus['gesture'] = "minus"
gesture_mul = pd.read_csv(r"resources/pose_index_mul.csv",index_col=False)
gesture_mul['gesture'] = "mul"
gesture_plus = pd.read_csv(r"resources/pose_index_plus.csv",index_col=False)
gesture_plus['gesture'] = "plus"
gesture_natual = pd.read_csv(r"resources/pose_index_natual.csv",index_col=False)
gesture_natual['gesture'] = "natual"

df = pd.concat([gesture_0, gesture_1, gesture_2,gesture_3,gesture_4,gesture_5,gesture_6,gesture_7,gesture_8,gesture_9,gesture_compute,gesture_delete,gesture_div,gesture_minus,gesture_mul,gesture_plus,gesture_natual], ignore_index=True)

df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("mixed.csv",index=False)

# %%



