import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from database.DataFrameDataBase import * 

# Initialize the database: 
study = StudyDataset.from_roots(
    roots=[
        ("PFC_lesion", "normal",
            "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/normal_mp"),
        ("PFC_lesion", "hallway_swap",
            "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/switching_halls"),
    ],
    animal_treatment_map_csv="/Users/ben/Documents/Data/lesion_study/animal_important_dates_csv.csv",
    filename_regex=r"(?P<animal>P\d{4,})_(?P<date>\d{4}_\d{2}_\d{2}).*\.csv$",
    # You may pass low_memory for the C engine; it will be auto-dropped for the Python engine:
    low_memory=False,
)

# Use helper function to get normal matching pennies data from the PFC_lesion experiment 
normal_df = study.all_trials(experiment="PFC_lesion", paradigm="normal")
