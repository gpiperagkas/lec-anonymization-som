# Copyright 2025 Grigorios Piperagkas
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


# Mondrian multidimensional partitioning k-anonymization algorithm
# Parts/Insights for this code have been supported by LLMs.
import pandas as pd
import numpy as np
from flask import Flask
import json

mondrian = Flask(__name__)

#Suppression function: replacing values with dummy character '*' for selected attributes.
def suppress_values(group,k):
    if len(group) <= k:
        return group.apply(lambda x: '*' if x.name in ["energy_consumption", "pv_production", "wind_turbine_production", "ev_charging", "energy_storage"] else x)
    else:
        return group


def mondrian_generalize_k(df, k):
    df_temp = df.copy()

    #For k<2 there is obviously error, in case of k=2 only mean values can replace input data, otherwise there is identification problem.
    if k < 2:
        raise ValueError('k must be greater than or equal to 2')

    for column in df_temp.columns:
        values = df_temp[column]
        #Find ranges with step according to min and max values of data group.
        if len(values) >= k:
            min_value = df_temp[column].min()
            max_value = df_temp[column].max()
            step = (max_value - min_value) / (k - 1)
            ranges = [(min_value + step * i, min_value + step * (i + 1)) for i in range(k)]
            ranges[-1] = (ranges[-1][0], max_value)

            # Create a dictionary to track generalized values
            generalized_values = {}

            for i, row in df_temp.iterrows():
                for start, end in ranges:
                    if row[column] >= start and row[column] <= end and len(generalized_values) < k:
                        df_temp.at[i, column] = f'{start:.1f}-{end:.1f}' # also mean value can be used as alternative: (start+end)/2. For statistical data exploitation, mean values are more suitable.
                        generalized_values[start] = generalized_values.get(start, 0) + 1
                        break

            # If fewer than k elements were generalized, revert the last generalized value
            if len(generalized_values) < k:
                last_generalized_start = list(generalized_values.keys())[-1]
                df_temp.loc[df_temp[column] == f'{last_generalized_start:.1f}-{max_value:.1f}', column] = df_temp[column].loc[df_temp[column] == f'{last_generalized_start:.1f}-{max_value:.1f}'].apply(lambda x: x.split('-')[0])

    return df_temp


def mondrian_median(data, k, current_column_index):

    #If the data group reaches k elements, generalize them.
    if len(data) == k:
        data_gen = mondrian_generalize_k(data, k)
        return data_gen
    #If the partitioning reaches less than k elements, apply suppression to the data group.
    if len(data) < k:
        sensitive_attributes = ["energy_consumption", "pv_production", "wind_turbine_production", "ev_charging", "energy_storage"]
        partition = data.groupby(sensitive_attributes).size().reset_index(name='counts')
        # Apply the suppression function to each group
        groups = partition.groupby(["energy_consumption", "pv_production", "wind_turbine_production", "ev_charging", "energy_storage"], group_keys=True)
        anonymized_data = pd.DataFrame()
        anonymized_data= groups.apply(suppress_values,k)
        return anonymized_data

    # For random partitioning, choose a random column to split on
    #column = random.choice(data.columns)

    # For deterministic partitioning, circulate the partitioning index to columns.
    if current_column_index < len(data.columns):
        column = data.columns[current_column_index]
        current_column_index += 1
    else:
        # Reset the index if all columns have been processed
        current_column_index = 0
        column = data.columns[current_column_index]

    # Find median split point
    data.sort_values(by=column)
    split = data[column].median()
    # Split the group into two
    left = data[data[column] < split]
    right = data[data[column] >= split]

    # Recursively anonymize both groups
    left_anonymous = mondrian_median(left, k,current_column_index)
    right_anonymous = mondrian_median(right, k,current_column_index)

    return pd.concat([left_anonymous, right_anonymous])



@mondrian.route('/')
def runapp():
  # Input data
  # Input data generation with random distributions, for each variable selected.
  # Categorical variables are not assumed in the experimentation, but can be added.
  energy_data = {
      "energy_consumption": np.random.normal(80, 20, 60),  # Reduced average consumption
      "pv_production": np.random.normal(30, 10, 60),  # Reduced PV production
      "wind_turbine_production": np.random.normal(25, 10, 60),  # Reduced wind turbine production
      "ev_charging": np.random.poisson(20, 60),  # Reduced EV charging
      "energy_storage": np.random.uniform(15, 40, 60)  # Reduced storage capacity
  }
  df = pd.DataFrame(energy_data)
  df[df < 0] = 0
  df_numeric= df
  #save input dataset.
  df_numeric.to_csv('input_anonymization.csv', index=False)

  # Select value of k parameter. Depends also on input data.
  k = 4

  anonymized_df = mondrian_median(df, k, 3)
  rounded_df = anonymized_df.round(decimals=1)

  # select quasi-identifiers according to application scenario
  quasi_identifiers = ["energy_consumption", "pv_production", "wind_turbine_production", "ev_charging", "energy_storage"]

  print(rounded_df)

  anonymized_df_json = rounded_df.to_json(orient='records')
  with open('anonymized_dataset_LEC.json', 'w') as f:
      json.dump(anonymized_df_json, f)
  anonymized_df.to_csv('anonymized_dataset_LEC.csv', index=False)
  return anonymized_df_json





