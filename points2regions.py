import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans as KMeans
from geojson import labelmask2geojson
from features import points2features
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import TSNE
from skimage.color import lab2rgb
from sklearn.metrics import pairwise_distances
from skimage.transform import rescale        
from typing import Optional, Any, Dict, List, Union, Literal, Tuple

def _compute_cluster_center(cluster_labels, features):
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels)

    # Initialize an array to store the sum of features for each cluster
    sum_features_per_cluster = np.zeros((num_clusters, features.shape[1]))

    # Count occurrences of each label
    label_counts = np.bincount(cluster_labels)

    # Accumulate the sum of features for each cluster
    for label, count in zip(unique_labels, label_counts[unique_labels]):
        mask = (cluster_labels == label)
        sum_features_per_cluster[label] = np.sum(features[mask], axis=0)

    # Compute the average features for each cluster
    average_features_per_cluster = sum_features_per_cluster / label_counts[unique_labels, None]

    return unique_labels, average_features_per_cluster


def _merge_clusters(kmeans, merge_thresh):
    linkage_matrix = linkage(kmeans.cluster_centers_, method='complete', metric='cosine')
    # fcluster returns in [1, n_clust], we prefer to start at 0 (hence -1).
    return  fcluster(linkage_matrix, merge_thresh, criterion='maxclust') - 1

def _rgb_to_hex(rgb):
    # Convert each channel value from float (0-1) to integer (0-255)
    r, g, b = [int(x * 255) for x in rgb]
    # Format the RGB values as a hexadecimal string
    hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
    return hex_color


class Points2Regions:
    """
    Points2Regions is a tool for clustering and defining regions based on categorical 
    marker data, which are commonly encountered in spatial biology.

    Methods:
        extract_features: Extracts features from input data.
        cluster: Performs clustering on the extracted features.
        get_cluster_colors: Retrieves cluster colors.
        get_labelmask: Generates label mask.
        get_anndata: Creates an AnnData object.
        get_geojson: Generates GeoJSON representation.
        get_clusters_per_marker: Retrieves clusters per marker.
        get_cluster_per_pixel: Placeholder method.
    """
        

    def __init__(
            self
        ):
        """
        Initializes Points2Regions instance.
        """
                
        self._features_extracted = False



    def extract_features(self, 
        xy:np.ndarray, 
        labels:np.ndarray, 
        pixel_width:float, 
        pixel_smoothing:float, 
        min_num_pts_per_pixel:float=0.0, 
        datasetids:Optional[np.ndarray]=None
    ):
        """
        Extracts features from input data.

        Args:
            xy (np.ndarray): Array of coordinates. Must be of shape (N x 2).
            labels (np.ndarray): Array of labels. Must be of shape (N).
            pixel_width (float): Width of the pixel.
            pixel_smoothing (float): How many pixels we should consider when smoothing.
            min_num_pts_per_pixel (float): Minimum number of points per pixel (after smoothing). 
            datasetids (Optional[np.ndarray]): Array of dataset ids. Must be of shape (N).
                If provided, features are independently computed for each unique
                dataset id.

        Returns:
            Points2Regions: Updated instance.

        Raises:
            ValueError: If input shapes are incompatible.
        """


        # Set clusters
        self._xy = np.array(xy, dtype='float32')

        # Set labels
        self._labels = labels
        self._unique_labels = np.unique(labels)
        
        # Set dataset ids
        self.datasetids = datasetids
        
         # Create list for slicing data by dataset id
        if self.datasetids is not None:
            unique_datasetids = np.unique(self.datasetids)
            iterdata = [
                (data_id, (
                    self._xy[self.datasetids==data_id,:],
                    self._labels[self.datasetids==data_id]
                )) for data_id in unique_datasetids
            ]
        else:
            iterdata = [('id', (self._xy, self._labels))]
        
        # Get features per dataset
        # Store in complicated dicionary
        self._results = {
            datasetid : points2features(
                xy_slice,
                labels_slice,
                self._unique_labels,
                pixel_smoothing,
                pixel_width,
                min_num_pts_per_pixel,
            )
            for datasetid, (xy_slice, labels_slice) in iterdata
        }

        self._features_extracted = True
        return self


    def cluster(self, num_clusters:int, seed:int=42):
        """
        Performs clustering on the extracted features.
        The method `extract_feature` must be called
        before calling `cluster`.

        The results can be extracted using the methods:
            - `get_clusters_per_marker` to get cluster per marker
            - `get_clusters_per_pixel` to get cluster for each pixel
            - `get_label_mask` to get the clusters as a label mask
            - `get_rgb_image` to get an RGB image where similar
                colors indicate similar clusters.
            - `get_geojson` to get the clusters as geojson polygons
            - `get_anndata` to the clusters in an anndata object 
            - `get_cluster_colors` to get colors for each cluster.
                Similar clusters will be colored similarly.
                
        Args:
            num_clusters (int): Number of clusters.
            seed (int): Random seed.

        Returns:
            Points2Regions: Updated instance.
        """

        # Create train features
        self.X_train = sp.vstack([
            result['features'][result['passed_threshold']] for result in self._results.values()
        ])

        # Run K-Means
        n_kmeans_clusters = int(1.3 * num_clusters)
        kmeans = KMeans(n_clusters=n_kmeans_clusters, n_init='auto', random_state=seed)
        kmeans = kmeans.fit(self.X_train)
       
        # Merge clusters using agglomerative clustering
        clusters = _merge_clusters(kmeans, num_clusters)

        # Compute new cluster centers
        _, self.cluster_centers = _compute_cluster_center(clusters, kmeans.cluster_centers_)
   
        # Iterate over datasets
        for datasetid, result_dict in self._results.items():
            
            # Get features and boolean indices for features passing threshold
            features, passed_threshold, pix2marker_ind = result_dict['features'], result_dict['passed_threshold'], result_dict['pix2marker_ind']
            
            # Get kmeans clusters for each dataset id
            kmeans_clusters = kmeans.predict(features[passed_threshold])

            # Get clusters per pixel
            merged_clusters = clusters[kmeans_clusters]
            cluster_per_pixel = np.zeros(features.shape[0], dtype='int') - 1
            cluster_per_pixel[passed_threshold] = merged_clusters

            # Get clusters per marker
            cluster_per_marker = cluster_per_pixel[pix2marker_ind]

            # Store result in a dictioanry
            self._results[datasetid]['cluster_per_marker'] = cluster_per_marker
            self._results[datasetid]['cluster_per_pixel'] = cluster_per_pixel

        self._num_points = len(self._xy)
        
        # Compute colors
        self._set_cluster_colors()

        return self



    def get_cluster_colors(self, hex:bool=False) -> np.ndarray:
        """
        Retrieves cluster colors.

        Args:
            hex (bool): Flag indicating whether to return colors in hexadecimal format.

        Returns:
            Union[np.ndarray, List[str]]: Cluster colors.
        """
        if not hex:
            return self.colors
        else:
            return np.array([_rgb_to_hex(rgb) for rgb in self.colors])
        
    def _set_cluster_colors(self):

        # If only one cluster, choose green
        if len(self.cluster_centers) == 1:
            self.colors = np.array([0, 1.0, 0]).reshape((1,-1))
            return
        
        # Compute distances between clusters
        D = pairwise_distances(self.cluster_centers)

        # Map each factor to a color
        embedding = TSNE(
            n_components=2, 
            perplexity=min(len(self.cluster_centers) - 1, 30), 
            init='random', 
            metric='precomputed',
            random_state=1
        ).fit_transform(
            D
        )

        # We interpret the 2D T-SNE points as points in a CIELAB space.
        # We then convert the CIELAB poitns to RGB.
        mu = 5
        out_ma = 128
        out_mi = -128
        mi, ma = np.percentile(embedding, q=mu, axis=0, keepdims=True), np.percentile(embedding, q=100-mu, axis=0, keepdims=True)
        colors = np.clip((embedding - mi) / (ma - mi), 0.0, 1.0)
        colors = (out_ma - out_mi) * colors + out_mi 
        colors = np.hstack((np.ones((len(colors),1))*70, colors))
        self.colors = lab2rgb(colors)
        self.colors = np.vstack((self.colors, np.zeros(3)))

    def get_labelmask(self, majority_radius:Optional[float]=None) -> np.ndarray:
        """
        Generates label mask.

        Args:
            majority_radius (Optional[float]): Fill pixels background pixels using
            majority voting within a region of radiu `majority_radius`.

        Returns:
            np.ndarray: Label mask.
        """
        masks = {}
        for id,result in self._results.items():
            
            grid_props = result['grid_props']
            
            # Create label mask
            clusters = result['cluster_per_pixel']
            label_mask = np.zeros(grid_props['grid_size'], dtype='uint8')
            label_mask[tuple(ind for ind in grid_props['grid_coords'])] = clusters + 1

            # Upscale the mask to match data
            upscale = 1.0 / grid_props['grid_scale']
            shift = grid_props['grid_offset'].astype('int') # yx
            label_mask = rescale(label_mask, upscale, order=0)
            new_label_mask = np.zeros((label_mask.shape[0] + shift[0],  label_mask.shape[1] + shift[1]), dtype='uint8')
            new_label_mask[shift[0]:shift[0]+label_mask.shape[0], shift[1]:shift[1]+label_mask.shape[1]] = label_mask
            masks[id] = label_mask.T

        if majority_radius is not None:
            from scipy.ndimage import distance_transform_edt as edt, binary_erosion
            from scipy.spatial import cKDTree as KDTree
            for id, mask in masks.items():

                    # Mask foreground from background
                    binary_mask = mask != 0

                    # Compute distance from each background pixel to foreground
                    distances = edt(~binary_mask)

                    # Get coordinates of background pixels that are close
                    # to foreground
                    yx_bg = np.vstack(np.where(distances < majority_radius)).T
                    yx_fg = np.vstack(np.where(binary_mask)).T
                    _, ind = KDTree(yx_fg).query(yx_bg, k=1)
                    ind = np.array(ind).flatten()
                    
                    mask[yx_bg[:,0], yx_bg[:,1]] = mask[yx_fg[ind,0], yx_fg[ind,1]]

                    # Erode to remove over-bluring near borders
                    binary_mask = mask != 0
                    binary_mask = binary_erosion(binary_mask, iterations=int(majority_radius), border_value=1)
                    mask[~binary_mask] = 0
                    masks[id] = mask

        if len(self._results.keys()) == 1:
            return masks[id]
        return masks


    def _create_anndata(self):
            
        # Create an adata object
        import anndata
        import pandas as pd

        # Get position of bins for each group (library id)
        xy_bin = np.vstack([
            r['xy_bin'][r['passed_threshold']] for r in self._results.values()
        ])

        # Get labels of bins for each group (library id)
        labels_bin = np.hstack([
            r['cluster_per_pixel'][r['passed_threshold']] for r in self._results.values()
        ])

        obs = {}
        obs['points2regions'] = labels_bin
        if len(self._results) > 1:
            obs['datasetid'] =  np.hstack([[id]*len(r['cluster_per_pixel']) for id, r in self._results.items()])

        # Multiply back features with the norm
        norms = 1.0 / np.hstack([r['norms'][r['passed_threshold']] for r in self._results.values()])
        norms[np.isinf(norms)] = 0
        norms = norms.reshape((-1,1))

        adata = anndata.AnnData(
            X=self.X_train.multiply(norms).tocsc(),
            obsm={
                'spatial' : xy_bin,
            },
            obs=obs,
            var=pd.DataFrame(index=self._unique_labels)
        )

        adata.obs['points2regions'] = adata.obs['points2regions'].astype('category')

        if len(self._results) > 1:
            adata.obs['datasetid'] = adata.obs['datasetid'].astype('category')

        # Create reads dataframe
        reads = {}
        reads['x'] = self._xy[:,0]
        reads['y'] = self._xy[:,1]
        reads['labels'] = self._labels
        reads['points2regions'] = self.get_clusters_per_marker()

        if self.datasetids is not None:
            reads['datasetid'] = self.datasetids

        reads = pd.DataFrame(reads).reset_index(drop=True)
        reads['labels'] = reads['labels'].astype('category')
        reads['points2regions'] = reads['points2regions'].astype('category')
        if self.datasetids is not None:
            reads['datasetid'] = reads['datasetid'].astype('category')
        adata.uns['reads'] = reads
        return adata


    def get_anndata(self) -> Any:
        """
        Creates an AnnData object.

        Returns:
            Any: AnnData object with:
                - marker-count vectors stored in `adata.X`.
                - Clusters stored in `adata.X`.
                - Position of each pixel stored in `adata.obsm["spatial"]`
                - Marker position and cluster per marker stored in `adata.uns`
        """
        return self._create_anndata()

    def get_geojson(self) -> Union[Dict, List]:
        """
        Generates GeoJSON representation of teh regions.

        Returns:
            Union[Dict, List]: GeoJSON data.
        """
        geojsons = {}
        for id, result in self._results.items():
            geojson = self._labelmask2geojson(result, region_name='My regions', colors=self.colors)
            geojsons[id] = geojson
        if len(geojsons) == 1:
            return geojsons[id]
        return geojsons

    def get_clusters_per_marker(self) -> np.ndarray:
        """
        Retrieves clusters per marker.

        Returns:
            np.ndarray: Clusters per marker.
        """
        cluster_per_marker = np.zeros(self._num_points, dtype='int')
        for datasetid in self._results.keys():
            cluster_per_marker[self._get_slice(datasetid)] = self._results[datasetid]['cluster_per_marker']
        return cluster_per_marker.copy()
    
    def get_cluster_per_pixel(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves clusters per marker and the location
        of each pixel

        Returns:
            Tuple[np.ndarray, np.ndarray]: Array with clusters and 
                array of pixel positions.
        """
        pass

    def _get_slice(self, datasetid):
        if self.datasetids is not None:
            return self.datasetids == datasetid
        else:
            return np.ones(len(self._xy), dtype='bool')

    def _labelmask2geojson(self, result, region_name, colors):
        grid_props = result['grid_props']
        clusters = result['cluster_per_pixel']
        label_mask = np.zeros(grid_props['grid_size'], dtype='uint8')
        label_mask[tuple(ind for ind in grid_props['grid_coords'])] = clusters+1
        geojson = labelmask2geojson(label_mask, region_name=region_name, scale=1.0/grid_props['grid_scale'], offset=grid_props['grid_offset'], colors=colors)
        return geojson






