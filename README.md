# Points2Regions

Points2Regions is a Python tool designed for clustering and defining regions based on categorical marker data, commonly encountered in spatial biology. It provides methods for feature extraction, clustering, and generating various outputs like label masks, GeoJSON representations, and more.

## Installation

You can install Points2Regions using the following command:

```bash
pip install git+https://github.com/wahlby-lab/Points2Regions
```

## Usage

```python
from points2regions import Points2Regions
import pandas as pd

# Example usage with a CSV file
data = pd.read_csv('https://tissuumaps.dckube.scilifelab.se/private/Points2Regions/toy_data.csv')

# Create the clustering model
p2r = Points2Regions(
    data[['X', 'Y']], 
    data['Genes'], 
    pixel_width=1, 
    pixel_smoothing=5
)

# Cluster with a specified number of clusters
p2r.fit(num_clusters=15)

# Get cluster label for each marker
cluster_per_marker = p2r.predict(output='marker')

# Get a label mask
label_mask, _ = p2r.predict(output='pixel')

# Get connected components
_, _, connected_components, num_components = p2r.predict(output='component')
```

## Example
See the Jupyter Notebook `example.ipynb` for examples.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


