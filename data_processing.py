import json

import pandas as pd

# load data in JSONs to dataframes
df_users = pd.read_json("JSONs/users_raw.json")
df_attendance = pd.read_json("JSONs/attendance_raw.json")

# explode/flatten the "times" in attendance JSON
df_exploded = df_attendance.explode("times").reset_index(drop=True)
times_df = pd.json_normalize(df_exploded["times"])

df_attendance = pd.concat([df_exploded.drop(columns=["times"]), times_df], axis=1)


# sample if the flattening worked, in JSON format
sample_flattened = df_attendance.iloc[0].to_dict()
with open("JSONs/sample_flattened.json", "w", encoding="utf-8") as f:
    json.dump(sample_flattened, f, indent=2, ensure_ascii=False, default=str)


# inspect it in the dataframe column structure...
print(df_attendance[["date", "user", "from", "to", "type", "wage"]].head(5))
