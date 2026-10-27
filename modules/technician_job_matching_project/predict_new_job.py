from pathlib import Path
import pandas as pd
B=Path(__file__).resolve().parent
df=pd.read_csv(B/"outputs"/"sample_new_job_rankings.csv")
print(df[["rank","technician_id","predicted_suitability_score","relevant_skill","inventory_match","distance_km","available_in_minutes"]].head(10).to_string(index=False))
