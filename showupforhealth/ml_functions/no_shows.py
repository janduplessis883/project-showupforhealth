from showupforhealth.params import *
import pandas as pd

def make_no_shows():
    print('\n=== Making No_shows_database =============================')
    data = pd.read_csv(f'{OUTPUT_DATA}/full_train_data.csv')
    noshows = data[['Patient ID', 'No_shows']].astype('int')
    unique_ids = noshows.drop_duplicates(keep='first')
    unique_ids.to_csv(f'{OUTPUT_DATA}/no_shows_db.csv', index=False)
    print(f'💾 Saved to output-data no_shows_db.csv - Shape: {unique_ids.shape}')

if __name__ == "__main__":
    make_no_shows()
    