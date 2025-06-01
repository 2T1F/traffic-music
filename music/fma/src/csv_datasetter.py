import pandas as pd

# ─── 0. CONFIGURATION: adjust these filenames to match your actual files ────

# (A) The CSV you already have that contains track_id, parsed_genre_ids, parsed_genre_titles, etc.
matched_csv = "C:\\Users\\OZBERK\\Desktop\\traffic-music\\music\\fma\\matched_tracks.csv"

# (B) The large echonest CSV (with a three‐row header) that contains all the raw audio_features.
echonest_csv = "C:\\Users\\OZBERK\\Desktop\\traffic-music\\music\\fma\\metadata\\raw_echonest.csv"

# (C) Output file where we’ll write the final merge
output_csv = "C:\\Users\\OZBERK\\Desktop\\traffic-music\\music\\fma\\matched_with_echonest_features.csv"
# ─── 1. LOAD YOUR EXISTING “MATCHED” CSV ─────────────────────────────────────
""" print("1) Loading matched_tracks.csv …")
matched_df = pd.read_csv(matched_csv)

if "track_id" not in matched_df.columns:
    raise KeyError(
        f"'track_id' not found in {matched_csv}. Columns are:\n{matched_df.columns.tolist()}"
    )
matched_df["track_id"] = matched_df["track_id"].astype(int)

print(f"→ Loaded {len(matched_df)} rows from 'matched_tracks.csv'.\n")


# ─── 2. READ THE ECHONEST CSV WITH A THREE‐ROW HEADER ─────────────────────────
print("2) Loading raw_echonest.csv with header=[0,1,2] …")
df_echonest = pd.read_csv(echonest_csv, header=[0, 1, 2], low_memory=False)

print(f"→ Echonest DataFrame shape: {df_echonest.shape}")
print("→ First few MultiIndex columns:")
print(df_echonest.columns[:8], "\n")


# ─── 3. EXTRACT FIRST COLUMN AS track_id + THE FIVE AUDIO_FEATURES ──────────
# 3.1. Treat the very first MultiIndex column as 'track_id'
track_col = df_echonest.columns[0]
print("→ Treating first column as track_id:", track_col)

# 3.2. Find the five under ("echonest","audio_features",<feature>)
features_to_extract = ["acousticness", "danceability", "energy",
                       "instrumentalness", "liveness"]

wanted_cols = [track_col]
for col in df_echonest.columns:
    lvl0, lvl1, lvl2 = col
    if (str(lvl0).strip().lower() == "echonest" and
        str(lvl1).strip().lower() == "audio_features" and
        lvl2 in features_to_extract):
        wanted_cols.append(col)

print("→ Will keep echonest columns:")
for tup in wanted_cols:
    print("   ", tup)
print()

# 3.3. Slice out just those columns
df_selected = df_echonest.loc[:, wanted_cols].copy()

# 3.4. Rename columns: first one → "track_id", others → lvl2
new_column_names = ["track_id"]
for col in wanted_cols[1:]:
    new_column_names.append(col[2])  # e.g. "acousticness"
df_selected.columns = new_column_names

print("→ df_selected columns after renaming:", df_selected.columns.tolist())

# 3.5. Drop the first row, which holds the textual "track_id" instead of a number
df_selected = df_selected.drop(index=0).reset_index(drop=True)

# 3.6. Now cast track_id to integer
df_selected["track_id"] = df_selected["track_id"].astype(int)

print("→ First 5 rows of df_selected (after dropping header‐row):")
print(df_selected.head(), "\n")


# ─── 4. MERGE matched_df ⇆ df_selected ON track_id ───────────────────────────
print("4) Merging matched_df with echonest features on 'track_id' …")
merged = pd.merge(
    matched_df,
    df_selected,
    on="track_id",
    how="inner"
)

print(f"→ After merge, {len(merged)} rows remain.")
print("→ Columns in merged DataFrame:")
print(merged.columns.tolist(), "\n")

print("→ Preview of merged data (first 5 rows):")
print(merged.head(), "\n")


# ─── 5. SAVE THE FINAL CSV ───────────────────────────────────────────────────
print(f"5) Saving the merged CSV to '{output_csv}' …")
merged.to_csv(output_csv, index=False)
print("→ Done. Final file written.") """

import pandas as pd

# 1) Load the merged CSV
full_df = pd.read_csv("C:\\Users\\OZBERK\\Desktop\\traffic-music\\music\\fma\\matched_with_echonest_features.csv")

# 2) List just the columns we want to keep
keep_cols = [
    "track_id",          # path to the audio (used at load time)
    "parsed_genre_ids",    # e.g. "21,10"
    "parsed_genre_titles", # e.g. "Hip-Hop,Pop"
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
]

# 3) Drop everything else
pruned_df = full_df[keep_cols].copy()

# 4) Display the result
print("Kept columns:", pruned_df.columns.tolist())
print("First few rows:")
print(pruned_df.head())

# 5) Save to a new “pruned” CSV
pruned_df.to_csv("C:\\Users\\OZBERK\\Desktop\\traffic-music\\music\\fma\\matched_pruned_for_training.csv", index=False)
print("→ Saved pruned CSV to 'matched_pruned_for_training.csv'")
