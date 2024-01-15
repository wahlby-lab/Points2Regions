import numpy as np
from skimage.measure import approximate_polygon, find_contours
from scipy.ndimage import zoom

COLORS = [
    [0.9019607843137255, 0.09803921568627451, 0.29411764705882354],
    [0.23529411764705882, 0.7058823529411765, 0.29411764705882354],
    [1.0, 0.8823529411764706, 0.09803921568627451],
    [0.2627450980392157, 0.38823529411764707, 0.8470588235294118],
    [0.9607843137254902, 0.5098039215686274, 0.19215686274509805],
    [0.5686274509803921, 0.11764705882352941, 0.7058823529411765],
    [0.27450980392156865, 0.9411764705882353, 0.9411764705882353],
    [0.9411764705882353, 0.19607843137254902, 0.9019607843137255],
    [0.7372549019607844, 0.9647058823529412, 0.047058823529411764],
    [0.9803921568627451, 0.7450980392156863, 0.7450980392156863],
    [0.0, 0.5019607843137255, 0.5019607843137255],
    [0.9019607843137255, 0.7450980392156863, 1.0],
    [0.6039215686274509, 0.38823529411764707, 0.1411764705882353],
    [1.0, 0.9803921568627451, 0.7843137254901961],
    [0.5019607843137255, 0.0, 0.0],
    [0.6666666666666666, 1.0, 0.7647058823529411],
    [0.5019607843137255, 0.5019607843137255, 0.0],
    [1.0, 0.8470588235294118, 0.6941176470588235],
    [0.0, 0.0, 0.4588235294117647],
    [0.5019607843137255, 0.5019607843137255, 0.5019607843137255],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0],
]

COLORS = [[int(255 * v) for v in RGB] for RGB in COLORS]



def polygons2json(polygons, cluster_class, cluster_names, colors=None):
    jsonROIs = []
    for i, polygon in enumerate(polygons):
        name = cluster_names[i]
        jsonROIs.append(
            {
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": []},
                "properties": {
                    "name": name,
                    "classification": {"name": cluster_class},
                    "color": colors[i] if colors is not None else [255, 0, 0],
                    "isLocked": False,
                },
            }
        )
        jsonROIs[-1]["geometry"]["coordinates"].append(polygon)
    return jsonROIs


def binarymask2polygon(binary_mask: np.ndarray, tolerance: float=0, offset: float=None, scale: float=None):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """

    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    binary_mask = zoom(binary_mask, 3, order=0, grid_mode=True)
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = find_contours(padded_binary_mask, 0.5)
    contours = [c-1 for c in contours]
    #contours = np.subtract(contours, 1)
    for contour in contours:
        contour = approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = contour / 3
        contour = np.rint(contour)
        if scale is not None:
            contour = contour * scale
        if offset is not None:
            contour = contour + offset  # .ravel().tolist()

        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        polygons.append(contour.tolist())

    return polygons



def labelmask2geojson(
    labelmask: np.ndarray,
    region_name:str="My regions",
    scale:float=1.0,
    offset:float = 0,
    colors=None
):
    from skimage.measure import regionprops

    nclusters = np.max(labelmask)
    if colors is None:
        colors = [COLORS[k % len(COLORS)] for k in range(nclusters)]
    if isinstance(colors, np.ndarray):
        colors = np.round(colors*255).astype('uint8')
        colors = colors.tolist()
    # Make JSON
    polygons = []
    cluster_names = [f"Region {l}" for l in np.arange(nclusters)]
    props = regionprops(labelmask+1)
    cc = []
    for index, region in enumerate(props):
        # take regions with large enough areas
        contours = binarymask2polygon(
            region.image,
            offset=scale*np.array(region.bbox[0:2]) + offset,
            scale=scale
        )
        cc.append(colors[region.label-1])
        polygons.append(contours)
    json = polygons2json(polygons, region_name, cluster_names, colors=cc)
    return json

