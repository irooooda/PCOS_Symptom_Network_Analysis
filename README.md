PCOS Symptom Network Analysis Pipeline
======================================

🎯 Project Objective
-------------------
This project builds an integrated analytical pipeline that transforms raw clinical data from PCOS (Polycystic Ovary Syndrome) patients into meaningful network-based insights. By analyzing symptom co-occurrence patterns, the project helps researchers uncover hidden relationships, community structures, and potential subtypes within PCOS symptomatology.

📁 Project Structure
--------------------
- config.yaml  
  Central configuration file specifying data paths, filenames, threshold values, network settings, and visualization parameters.

- PCOS_data_without_infertility.xlsx  
  Raw clinical dataset of PCOS patients (excluding infertility cases).

- data_cleaning.py  
  Preprocesses the raw dataset by handling missing data, cleaning columns, and transforming or binarizing features as defined in config.yaml.

- symptom_coocurence.py  
  Computes symptom co-occurrence matrices by analyzing how frequently symptoms appear together among patients.

- network_utils.py  
  Contains helper functions for network construction, filtering, and computing network statistics to support the visualization and analysis processes.

- symptom_network_visuals.py  
  Generates various visualizations of the symptom network including static images, interactive HTML-based networks, heatmaps, and community detection plots.

- Output Files (generated during execution):  
  - cleaned_pcos_data.csv and binary_transformed_pcos_data.csv  
  - Co-occurrence matrices  
  - Visualizations: network_static.png, interactive_network.html, heatmap.png, and communities.png

⚙️ How to Run
-------------
1. Environment Setup:  
   Ensure you have Python 3.x installed, along with the required libraries. Install dependencies using:
   pip install pandas numpy matplotlib networkx pyyaml openpyxl plotly python-louvain scipy
   *(Adjust the package list as needed if any additional dependencies are required.)*

2. File Setup:  
   Place all project files (i.e., config.yaml, PCOS_data_without_infertility.xlsx, and the Python scripts) in your working directory, ensuring that directory paths match those specified in the configuration file.

3. Configuration Adjustments:  
   Modify config.yaml if necessary—for example, update file paths, threshold values, or visualization settings.

4. Execution Order:  
   Run the scripts in the following sequence:
   python data_cleaning.py
   python symptom_coocurence.py
   python symptom_network_visuals.py

🧠 Methodology
--------------
- Data Cleaning:  
  The data_cleaning.py script processes the raw dataset by removing columns with excessive missing values, handling missing data beyond a 50% threshold, and transforming or binarizing relevant features as per the configuration settings.

- Symptom Co-occurrence Analysis:  
  The symptom_coocurence.py script calculates pairwise symptom relationships to generate co-occurrence matrices, quantifying how frequently symptoms appear together across the patient records.

- Network Construction:  
  Utilizing helper functions from network_utils.py, the pipeline filters and constructs a network graph based on co-occurrence data. Settings such as a minimum edge weight and the top N nodes (defined in config.yaml) ensure that only significant relationships are visualized.

- Visualization & Community Detection:  
  The symptom_network_visuals.py script produces:
    - A static overview image of the symptom network,
    - An interactive HTML version for in-depth exploration,
    - A heatmap showing the intensity of symptom co-occurrences, and
    - Community detection visualizations (using the Louvain algorithm) to highlight clusters within the network.

📈 Output Visuals
-----------------
- network_static.png:  
  A static image capturing the overall symptom relationship network.
  
- interactive_network.html:  
  An interactive, zoomable network visualization for detailed exploration.

- heatmap.png:  
  A heatmap that visualizes the intensity and frequency of symptom co-occurrences.

- communities.png:  
  A visual representation of community structures within the network, revealing potential clusters of related symptoms.

💡 Future Improvements
------------------------------
- Integrate all pipeline steps into a unified CLI or main script for streamlined execution.
- Enhance logging and error handling to increase robustness.
- Develop additional visualizations and statistical summaries.
- Containerize the application using Docker or virtual environments to ensure reproducibility.
- Extend the analysis to incorporate additional clinical data or multi-omics datasets for comprehensive research applications.

📚 Credits
----------
Created by Iroda Ulmasboeva as part of a Network Science and Telematics Lab.
For questions/collaboration: ulmasboevairoda@gmail.com