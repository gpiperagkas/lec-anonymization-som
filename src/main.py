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

# SOM training and visualization, k-anonymization for energy data in Local energy communities, January 2025

import sys

from mondrian_k_anonymization import mondrian
from som_single import generate_input,train_som, plot_results
from som_LEC import generate_input_LEC, train_som_LEC, plot_results_LEC

if __name__ == '__main__':

    # Check if a command-line argument is provided
    if len(sys.argv) > 1:
        # Get the argument
        operation = sys.argv[1]
    else:
        print("Please provide input operation: anonymization or som")
        sys.exit(0)  # Exit

    if (operation == "anonymization"):
        # run flask app
        mondrian.run(debug=True)
    elif (operation == "som"):
        # Run SOM training for both cases
        # Generate input data, train som and plot figures for one metering device, for 30 days period.
        data = generate_input()
        # grid dimensions for SOM hardcoded to som_x = 10 and som_y = 10, change accordingly for experimentation.
        som, flattened_weights, data_normalized = train_som(data)
        plot_results(som, 10, 10, flattened_weights, data, data_normalized)
        # Generate input data, train som and plot figures for 60 metering devices of a LEC, for 30 days period.
        data_LEC = generate_input_LEC()
        som_LEC, flattened_weights_LEC, data_normalized_LEC = train_som_LEC(data_LEC)
        plot_results_LEC(som_LEC, 10, 10, flattened_weights_LEC, data_normalized_LEC)


