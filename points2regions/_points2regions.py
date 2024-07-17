from typing import Optional, Any, Dict, List, Union, Tuple, Literal
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from skimage.color import lab2rgb
from scipy.spatial import cKDTree as KDTree
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.ndimage import distance_transform_edt as edt, binary_erosion

from .geojson import labelmask2geojson
from .utils import inverse_distance_interpolation, attribute_matrix


def _compute_cluster_center(cluster_labels, features):
    A, unique_labels = attribute_matrix(cluster_labels)
    average_features_per_cluster = ((A.T @ features) / (A.T.sum(axis=1))).A
    return unique_labels, average_features_per_cluster


def _merge_clusters(kmeans, merge_thresh):
    linkage_matrix = linkage(kmeans.cluster_centers_, method='complete', metric='cosine')
    # fcluster returns in [1, n_clust], we prefer to start at 0 (hence -1).
    return  fcluster(linkage_matrix, merge_thresh, criterion='maxclust') - 1

def _rgb_to_hex(rgb):
    # Convert each channel value from float (0-1) to integer (0-255)
    r, g, b = [int(x * 255) for x in rgb]
    # Format the RGB values as a hexadecimal string
    hex_color = f"#{r:02X}{g:02X}{b:02X}"
    return hex_color


class Points2Regions:
    """
    Points2Regions is a tool for clustering and defining regions based on categorical 
    marker data, which are commonly encountered in spatial biology.
    
    
    """

    def __init__(self, xy:np.ndarray, labels:np.ndarray, pixel_width:float, pixel_smoothing:float, min_num_pts_per_pixel:float=0.0, datasetids:Optional[np.ndarray]=None):
        """
        Initializes Points2Regions instance.

        Parameters
        ----------
        xy : np.ndarray
            Array of coordinates. Must be of shape (N x 2).
        labels : np.ndarray
            Array of labels. Must be of shape (N).
        pixel_width : float
            Width of a pixel in the rasterized image.
        pixel_smoothing : float
            How many pixels we should consider when smoothing.
        min_num_pts_per_pixel : float, optional
            Minimum number of points per pixel (after smoothing).
        datasetids : np.ndarray, optional
            Array of dataset ids. Must be of shape (N).
            If provided, features are independently computed for each unique dataset id.

        """             
        self._features_extracted = False
        self._xy = None
        self._unique_labels = None
        self._datasetids = None
        self._results = {}
        self._labels = None
        self._colors = None
        self._num_points = None
        self.cluster_centers = None
        self._is_clustered = False
        self.inertia = None
        self._extract_features(xy, labels, pixel_width, pixel_smoothing, min_num_pts_per_pixel, datasetids)


    def fit(self, num_clusters:int, kmeans_kwargs:Optional[Dict[str,Any]]=None, seed:int=42):
        """
        Fit the clustering model on the extracted features.

        Parameters
        ----------
        num_clusters : int
            Number of clusters to form.
        seed : int
            Seed for random initialization.
        kmeans_kwargs : dict, optional
            Dictionary with keyword arguments to be passed to 
            scikit-learn's MiniBatchKMeans constructor. 
        """
                
        self._cluster(num_clusters, seed, kmeans_kwargs)
        return self



    def predict(self, output: Literal['marker', 'pixel', 'anndata','geojson', 'colors', 'connected'], adata_cluster_key:str='Clusters', grow:Optional[int]=None, min_area:int=1) -> Any:
        """
        Fit and predict the output based on the specified format after fitting the clustering model.

        Parameters
        ----------
        num_clusters : int
            Number of clusters to form.

        output : Literal['marker', 'pixel', 'anndata','geojson', 'colors'] specifying the output format, where

            * marker : np.ndarray 
                - Returns the cluster label for each marker as an ndarray. Clusters equal to -1 correspond to background.
            
            * pixel : Tuple(np.ndarray, np.ndarray)
                - Returns a label mask of size `height` times `width` where each pixel is labelled by a cluster.
                - Also returns the parameters, `T`, such that input markers, `xy`, are mapped onto the label mask via `T[0] * xy + T[1]`.
                - The optional parameter `grow` can be set to an integer value for growing the label mask's foreground pixels.
            
            * anndata : AnnData, returns an AnnData object with:
                - Marker-count vectors stored in `adata.X`.
                - Clusters stored in `adata.obs[adata_cluster_key]`.
                - Position of each pixel stored in `adata.obsm["spatial"]`.
                - Marker position and cluster per marker stored in `adata.uns`.
                
                Requires that the package `anndata` is installed.

            * geojson : Dict | List:
                - Returns a dictionary or list with geojson polygons

            * colors : np.ndarray
                - An ndarray of colors (hex) for each cluster. Similar clusters will have similar colors. Last entry correspond to the background color. 

            * connected : Tuple[np.ndarray, int, np.ndarray, np.ndarray], returns connected components in the label mask.
                Output comes as a tuple with four values:
                    - `connected` is a label mask where each connected component is uniquely labelled.
                    - `num_components` is an integer indicating the number of unique connected components.
                    - `label_mask` is the label mask
                    - `tform` contains the slope and instersect so that input markers' positions, `xy`, can be mapped onto the label mask via `tform[0] * xy + tform[1]`
                The optional parameter, `grow`, can be used to grow the size of the label mask.
                The optional parameter, `min_area`, can be used to remove small connected components in the geojson polygons
                    or in the connected component label mask. 
        
        seed : int, optional
            Random seed for clustering.

        adata_cluster_key : str, optional
            Key indicating which column to store the cluster ids in an AnnData object (default: 'Clusters').
        
        grow : Optional[int], optional
            If provided, the number of pixels to grow the foreground regions in the label mask.
                                                
        Returns
        -------
        Points2Regions
            Updated instance of the Points2Regions class.

        """



        if not self._is_clustered:
            raise ValueError('Must run the method `.fit(...)` before `.predict(...)`, or use `.fit_predict(...)`')
        if output == 'marker':
            return self._get_clusters_per_marker()
        elif output == 'pixel':
            return self._get_labelmask(grow=grow)
        elif output == 'anndata':
            return self._get_anndata(cluster_key_added=adata_cluster_key)
        elif output == 'geojson':
            return self._get_geojson(grow=grow, min_area=min_area)
        elif output == 'colors':
            return self._get_cluster_colors(hex=True)
        elif output == 'connected':
            label_mask_args = self._get_labelmask(grow=grow)
            return self._get_connected_components(label_mask_args, min_area=min_area)
        else:
            valid_inputs = {'marker', 'pixel', 'anndata','geojson', 'colors', 'connected'}
            raise ValueError(f'Invalid value for `output` {output}. Must be one of the following: {valid_inputs}.')
    
    def fit_predict(self, num_clusters:int, output: Literal['marker', 'pixel', 'anndata','geojson', 'colors', 'connected'], seed:int=42, kmeans_kwargs:Optional[Dict[str, Any]]=None, adata_cluster_key:str='Clusters', grow:Optional[int]=None, min_area:int=1) -> Any:
        """
        Fit and predict the output based on the specified format after fitting the clustering model.

        Parameters
        ----------
        num_clusters : int
            Number of clusters to form.

        output : Literal['marker', 'pixel', 'anndata','geojson', 'colors'] specifying the output format, where

            * marker : np.ndarray 
                - Returns the cluster label for each marker as an ndarray. Clusters equal to -1 correspond to background.
            
            * pixel : Tuple(np.ndarray, np.ndarray)
                - Returns a label mask of size `height` times `width` where each pixel is labelled by a cluster.
                - Also returns the parameters, `T`, such that input markers, `xy`, are mapped onto the label mask via `T[0] * xy + T[1]`.
                - The optional parameter `grow` can be set to an integer value for growing the label mask's foreground pixels.
            
            * anndata : AnnData, returns an AnnData object with:
                - Marker-count vectors stored in `adata.X`.
                - Clusters stored in `adata.obs[adata_cluster_key]`.
                - Position of each pixel stored in `adata.obsm["spatial"]`.
                - Marker position and cluster per marker stored in `adata.uns`.
                
                Requires that the package `anndata` is installed.

            * geojson : Dict | List:
                - Returns a dictionary or list with geojson polygons

            * colors : np.ndarray
                - An ndarray of colors (hex) for each cluster. Similar clusters will have similar colors. Last entry correspond to the background color. 

            * connected : Tuple[np.ndarray, int, np.ndarray, np.ndarray], returns connected components in the label mask.
                Output comes as a tuple with four values:
                    - `connected` is a label mask where each connected component is uniquely labelled.
                    - `num_components` is an integer indicating the number of unique connected components.
                    - `label_mask` is the label mask
                    - `tform` contains the slope and instersect so that input markers' positions, `xy`, can be mapped onto the label mask via `tform[0] * xy + tform[1]`
                The optional parameter, `grow`, can be used to grow the size of the label mask.
                The optional parameter, `min_area`, can be used to remove small connected components in the geojson polygons
                    or in the connected component label mask. 
        
        seed : int, optional
            Random seed for clustering.

        kmeans_kwargs : dict, optional
            Dictionary with keyword arguments to be passed to 
            scikit-learn's MiniBatchKMeans constructor. 

        adata_cluster_key : str, optional
            Key indicating which column to store the cluster ids in an AnnData object (default: 'Clusters').
        
        grow : Optional[int], optional
            If provided, the number of pixels to grow the foreground regions in the label mask.
                                                
        Returns
        -------
        Points2Regions
            Updated instance of the Points2Regions class.

        """
       
        self.fit(num_clusters, kmeans_kwargs, seed)
        return self.predict(output, adata_cluster_key, grow, min_area)

    def _extract_features(self, xy:np.ndarray, labels:np.ndarray, pixel_width:float, pixel_smoothing:float, min_num_pts_per_pixel:float=0.0, datasetids:Optional[np.ndarray]=None):
        """
        Extracts features from input data.

        Parameters
        ----------
        xy : np.ndarray
            Array of coordinates. Must be of shape (N x 2).
        labels : np.ndarray
            Array of labels. Must be of shape (N).
        pixel_width : float
            Width of a pixel in the rasterized image.
        pixel_smoothing : float
            How many pixels we should consider when smoothing.
        min_num_pts_per_pixel : float, optional
            Minimum number of points per pixel (after smoothing).
        datasetids : np.ndarray, optional
            Array of dataset ids. Must be of shape (N).
            If provided, features are independently computed for each unique dataset id.

        Returns
        -------
        Points2Regions
            Updated instance.

        Raises
        ------
        ValueError
            If input shapes are incompatible.
        """

        # Set clusters
        self._xy = np.array(xy, dtype='float32')

        # Set labels
        self._labels = labels
        self._unique_labels = np.unique(labels)
        # Set dataset ids
        self._datasetids = datasetids
         # Create list for slicing data by dataset id
        if self._datasetids is not None:
            unique_datasetids = np.unique(self._datasetids)
            iterdata = [
                (data_id, (
                    self._xy[self._datasetids==data_id,:],
                    self._labels[self._datasetids==data_id]
                )) for data_id in unique_datasetids
            ]
        else:
            iterdata = [('id', (self._xy, self._labels))]
        
        # Get features per dataset
        # Store in complicated dicionary
        self._results = {}
        for datasetid, (xy_slice, labels_slice) in iterdata:
            self._results[datasetid] = inverse_distance_interpolation(
                    xy_slice,
                    labels_slice,
                    self._unique_labels,
                    pixel_width,
                    pixel_smoothing,
                    min_num_pts_per_pixel
                )

            self._features_extracted = True
        return self


    def _cluster(self, num_clusters:int, seed:int=42, kmeans_kwargs:Optional[Dict[str,Any]]=None):
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
                
        Parameters
        ----------
        num_clusters : int
            Number of clusters.
        seed : int, optional
            Random seed.
        kmeans_kwargs : dict, optional
            Dictionary with keyword arguments to be passed to 
            scikit-learn's MiniBatchKMeans constructor. 

        Returns
        -------
        Points2Regions
            Updated instance.
        """

        # Create train features
        self.X_train = sp.vstack([
            result['features'][result['passed_threshold']] for result in self._results.values()
        ])

        default_kmeans_kwargs = dict(
            init="k-means++",
            max_iter=100,
            batch_size=1024,
            verbose=0,
            compute_labels=True,
            random_state=seed,
            tol=0.0,
            max_no_improvement=10,
            init_size=None,
            n_init="auto",
            reassignment_ratio=0.005,
        )

        if kmeans_kwargs is not None:
            for key,val in kmeans_kwargs.items():
                if key in default_kmeans_kwargs:
                    default_kmeans_kwargs[key] = val

        if isinstance(default_kmeans_kwargs['init'], sp.spmatrix):
            n_kmeans_clusters = default_kmeans_kwargs['init'].shape[0]
            use_hierarchial = False
        elif isinstance(default_kmeans_kwargs['init'], np.ndarray):
            n_kmeans_clusters = default_kmeans_kwargs['init'].shape[0]
            use_hierarchial = False
        else:
            n_kmeans_clusters = int(1.5 * num_clusters)
            use_hierarchial = True

        kmeans = KMeans(
            n_kmeans_clusters,
            **default_kmeans_kwargs
        )

        kmeans = kmeans.fit(self.X_train)
        self.inertia = kmeans.inertia_

        # Merge clusters using agglomerative clustering
        if use_hierarchial:
            clusters = _merge_clusters(kmeans, num_clusters)
            # Compute new cluster centers
            _, self.cluster_centers = _compute_cluster_center(clusters, kmeans.cluster_centers_)
        else:
            clusters = kmeans.labels_
            self.cluster_centers = kmeans.cluster_centers_

   
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

        self._is_clustered = True

        return self



    def _get_cluster_colors(self, hex:bool=False) -> np.ndarray:
        """
        Retrieves cluster colors.

        Parameters
        ----------
        hex : bool, optional
            Flag indicating whether to return colors in hexadecimal format.

        Returns
        -------
        np.ndarray
            If hex is False, returns an ndarray of cluster colors. 
            If hex is True, returns an ndarray of cluster colors in hexadecimal format.
        """
        if not hex:
            return np.array(self._colors)
        else:
            return np.array([_rgb_to_hex(rgb) for rgb in self._colors])
        
    def _set_cluster_colors(self):


        # If only one cluster, choose green
        if len(self.cluster_centers) == 1:
            self._colors = np.array([0, 1.0, 0]).reshape((1,-1))
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
        self._colors = lab2rgb(colors)
        self._colors = np.vstack((self._colors, np.zeros(3)))

    def _get_labelmask(self, grow:Optional[float]=None) -> np.ndarray:
        """
        Generates label mask.

        Parameters
        ----------
        grow : float, optional
            Fill background pixels by growing foreground regions by a `grow` amount of pixels.

        Returns
        -------
        Tuple[np.ndarray, Tuple[float, float]]
            Label mask as an ndarray and transformation coefficients `T`,
            such that `xy * T[0] + T[1]` transforms the location of an
            input marker, `xy`, to the correct pixel in the label mask.
        """
        masks = {}
        for datasetid, result in self._results.items():
            
            grid_props = result['grid_props']
            
            # Create label mask
            clusters = result['cluster_per_pixel']
            label_mask = np.zeros(grid_props['grid_size'], dtype='int')
            label_mask[tuple(ind for ind in grid_props['grid_coords'])] = clusters + 1

            # Upscale the mask to match data
            scale = grid_props['grid_scale']
            shift = grid_props['grid_offset'] 
            T = (scale, -shift*scale)
            masks[datasetid] = (label_mask.T - 1, T)

        if grow is not None:
            for datasetid, (mask, T) in masks.items():

                    # Mask foreground from background
                    binary_mask = mask != -1

                    # Compute distance from each background pixel to foreground
                    distances = edt(~binary_mask)

                    # Get coordinates of background pixels that are close
                    # to foreground
                    yx_bg = np.vstack(np.where(distances < grow)).T
                    yx_fg = np.vstack(np.where(binary_mask)).T
                    _, ind = KDTree(yx_fg).query(yx_bg, k=1)
                    ind = np.array(ind).flatten()
                    
                    mask[yx_bg[:,0], yx_bg[:,1]] = mask[yx_fg[ind,0], yx_fg[ind,1]]

                    # Erode to remove over-bluring near borders
                    binary_mask = mask != -1
                    binary_mask = binary_erosion(binary_mask, iterations=int(grow), border_value=1)
                    mask[~binary_mask] = -1
                    masks[datasetid] = (mask, T)

        if len(self._results.keys()) == 1:
            return masks[datasetid]
        return masks


      


    def _get_anndata(self, cluster_key_added:str='Clusters') -> Any:
        """
        Creates an AnnData object.

        Parameters
        ----------
        cluster_key_added : str
            Key indicating which column to store the cluster ids in (default: `Clusters`)

        Returns
        -------
        Any
            AnnData object with:
                - Marker-count vectors stored in `adata.X`.
                - Clusters stored in `adata.X`.
                - Position of each pixel stored in `adata.obsm["spatial"]`.
                - Marker position and cluster per marker stored in `adata.uns`.
        """
        # Create an adata object
        import anndata
        import pandas as pd
        print('Creating anndata')
        # Get position of bins for each group (library id)
        xy_pixel = np.vstack([
            r['xy_pixel'][r['passed_threshold']] for r in self._results.values()
        ])

        # Get labels of bins for each group (library id)
        labels_pixel = np.hstack([
            r['cluster_per_pixel'][r['passed_threshold']] for r in self._results.values()
        ]).astype(str)

        obs = {}
        obs[cluster_key_added] = labels_pixel
        if len(self._results) > 1:
            obs['datasetid'] =  np.hstack([[id]*len(r['cluster_per_pixel'][r['passed_threshold']]) for id, r in self._results.items()])

        # Multiply back features with the norm
        norms = 1.0 / np.hstack([r['norms'][r['passed_threshold']] for r in self._results.values()])
        norms[np.isinf(norms)] = 0
        norms = norms.reshape((-1,1))

        # Remove the normalization
        X = self.X_train.multiply(norms).tocsc()

        # Remove the log transofrm.
        # X should be counts
        X.data = np.expm1(X.data)

        # Create the anndata object
        adata = anndata.AnnData(
            X=X,
            obs=obs,
            obsm=dict(spatial=xy_pixel)
        )

        adata.var_names = self._unique_labels
        adata.obs['datasetid'] = adata.obs['datasetid'].astype('int') 

        adata.obs[cluster_key_added] = adata\
            .obs[cluster_key_added]\
            .astype('category')

        if len(self._results) > 1:
            adata.obs['datasetid'] = adata\
                .obs['datasetid']\
                .astype('category')



        marker2pix_ind = []
        offset = 0
        for r in self._results.values():
            # Get indices of non empty bins
            non_empty = np.where(r['passed_threshold'])[0]

            # Remap each point in the dataset
            remap = {i : -1 for i in range(len(r['cluster_per_marker']))}
            for new, old in enumerate(non_empty):
                remap[old] = new + offset
            marker2pix_ind.append([remap[i] for i in r['pix2marker_ind']])
            offset += len(non_empty)
            
        marker2pix_ind = np.hstack(marker2pix_ind)

        
        # Create reads dataframe
        reads = {}
        reads['x'] = self._xy[:,0]
        reads['y'] = self._xy[:,1]
        reads['labels'] = self._labels
        reads[cluster_key_added] = self._get_clusters_per_marker()
        reads['pixel_ind'] = marker2pix_ind
        

        if self._datasetids is not None:
            reads['datasetid'] = self._datasetids
            reads['datasetid'] = reads['datasetid'].astype('int') 

        # Create the dataframe
        reads = pd.DataFrame(reads)

        # Change the datatypes
        reads['labels'] = reads['labels'].astype('category')
        reads[cluster_key_added] = reads[cluster_key_added].astype('category')
        if self._datasetids is not None:
            reads['datasetid'] = reads['datasetid'].astype('category')

        # Add to anndata
        adata.uns['reads'] = reads
        return adata

    def _get_geojson(self, grow:int=None, min_area:int=0) -> Union[Dict, List]:
        """
        Generates GeoJSON representation of the regions.

        

        Parameters
        ----------
        gorw : int
            The optional parameter, `grow`, can be used to grow the size of the label mask
            with this many pixels. Default 0.

        Returns
        -------
        Union[Dict, List]
            GeoJSON data.
        """

        # Get label mask and transformations
        if len(self._results) == 1:
            label_mask, tform = self._get_labelmask(grow=grow)
            geojson = labelmask2geojson(label_mask, scale=1.0 / tform[0], offset=-tform[1]/tform[0], min_area=min_area)
        else:
            geojson = {}
            for datasetid, (label_mask, tform) in self._get_labelmask(grow=grow):
                geojson[datasetid] = labelmask2geojson(label_mask, scale=1.0 / tform[0], offset=-tform[1]/tform[0], min_area=min_area)

        return geojson

    def _get_clusters_per_marker(self) -> np.ndarray:
        """
        Retrieves clusters per marker.

        Returns
        -------
        np.ndarray
            Clusters per marker.
        """
        cluster_per_marker = np.zeros(self._num_points, dtype='int')
        for datasetid, result in self._results.items():
            cluster_per_marker[self._get_slice(datasetid)] = result['cluster_per_marker']

        return cluster_per_marker.copy()


    def _get_slice(self, datasetid):
        if self._datasetids is not None:
            return self._datasetids == datasetid
        else:
            return np.ones(len(self._xy), dtype='bool')



    def _get_connected_components(self, label_mask_result: Union[Dict[str, Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]], min_area:int=1) -> Tuple[np.ndarray, int]:
        """
        Get connected components in the label mask.
        This can be slow for high-resolution masks,

        Parameters
        ----------
        label_mask : np.ndarray
            Label mask.
        min_area : int, optional
            Minimum area for connected components.

        Returns
        -------
        Tuple[np.ndarray, int, np.ndarray, Tuple[float,float]]
            Labels of connected components, the number of components, the label mask
            and a tuple of transformation parameters that can be used to map each 
            observed point onto the masks via `tform[0]*xy+tform[1]`
        """


        if not isinstance(label_mask_result, dict):
            dataset_dictionary = {
                'datasetid' : label_mask_result
            }
        else:
            dataset_dictionary = label_mask_result
                
        output = {}

        for datasetid, result in dataset_dictionary.items():
            # Get the shape of the label image
            label_mask = result[0]
            tform = result[1]

            # Shift label mask so that 0 is background instead of -1
            label_mask = label_mask + 1

            N, M = label_mask.shape
            total_pixels = N * M

            # Create 1D arrays for row and column indices of the adjacency matrix
            row_indices = np.arange(total_pixels).reshape(N, M)
            col_indices = np.copy(row_indices)


            # Mask for pixels with the same label in horizontal direction
            mask_same_label_horizontal = (label_mask[:, :-1] == label_mask[:, 1:]) & (label_mask[:, :-1] != 0)

            # Include connections between pixels with the same label
            row_indices_horizontal = row_indices[:, :-1][mask_same_label_horizontal].flatten()
            col_indices_horizontal = row_indices[:, 1:][mask_same_label_horizontal].flatten()

            # Mask for pixels with the same label in vertical direction
            mask_same_label_vertical = (label_mask[:-1, :] == label_mask[1:, :]) & (label_mask[:-1, :] != 0)

            # Include connections between pixels with the same label
            row_indices_vertical = col_indices[:-1, :][mask_same_label_vertical].flatten()
            col_indices_vertical = col_indices[1:, :][mask_same_label_vertical].flatten()

            # Combine the horizontal and vertical connections
            r = np.concatenate([row_indices_horizontal, row_indices_vertical])
            c = np.concatenate([col_indices_horizontal, col_indices_vertical])

            # Create COO format data for the sparse matrix
            data = np.ones_like(r)

            # Create the sparse matrix using COO format
            graph_matrix = sp.coo_matrix((data, (r, c)), shape=(total_pixels, total_pixels))

            # Remove duplicate entries in the COO format
            graph_matrix = sp.coo_matrix(graph_matrix)

            # Run connected component labelling
            num_components, labels = sp.csgraph.connected_components(graph_matrix, directed=False)

            # Compute frequency of each connected component
            counts = np.bincount(labels, minlength=num_components)

            # Mask out small components
            mask = counts<=min_area

            # Relabel labels such that
            # small components have label 0
            # other labels have labels 1->N+1
            counts[mask] = 0

            num_large_components = np.sum(~mask)
            counts[~mask] = np.arange(1, num_large_components+1)
            labels = counts[labels]
            

            # Shift so that labels are in [0, num_large_components)
            labels = labels - 1

            # Reshape to a grid
            labels = labels.reshape(label_mask.shape)

            output[datasetid] = (labels, num_large_components, label_mask, tform)

        if len(output) == 1:
            return output[datasetid]
        else:
            return output



