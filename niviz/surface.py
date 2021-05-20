"""
Provides utility functions to access Surface data
"""

import numpy as np
from enum import Enum


class INTENT(Enum):
    NIFTI_INTENT_POINTSET = 1008


def gifti_get_mesh(gifti):
    '''
    Extract vertices and triangles from GIFTI surf.gii
    file
    Arguments:
        gifti (GiftiImage): Input GiftiImage
    '''

    v, t = gifti.agg_data(('pointset', 'triangle'))
    return v.copy(), t.copy()


def gifti_get_full_brain_mesh(l_gifti, r_gifti):
    '''
    Construct a full brain mesh by joining
    both hemispheres

    Arguments:
        l_gifti: Left hemisphere GiftiImage
        r_gifti: Right hemisphere GiftiImage
    '''
    l_vert, l_trig = gifti_get_mesh(l_gifti)
    r_vert, r_trig = gifti_get_mesh(r_gifti)

    offset = l_trig.max() + 1
    r_trig += offset

    verts = np.vstack((l_vert, r_vert))
    trigs = np.vstack((l_trig, r_trig))

    return (verts, trigs, offset)


def map_cifti_to_giftis(l_gifti, r_gifti, cifti):
    '''
    Maps cifti data-array to gifti vertices to account
    for missing indices (i.e removal of medial wall)

    Arguments:
        l_gifti:    Left hemisphere GIFTI mesh
        r_gifti:    Right hemisphere GIFTI mesh
        cifti:      CIFTI file to map [Series x BrainModel]

    Returns:
        verts:          Vertices of surface mesh
        trigs:          Triangles of surface mesh
        mapping_array:  An [Features x Vertices] mapping array pulled
                        from the CIFTI image
    '''

    # Validate and obtain CIFTI indices
    cifti_vertices = None
    for mi in cifti.header.mapped_indices:
        map_type = cifti.header.get_index_map(mi).indices_map_to_data_type
        if map_type == "CIFTI_INDEX_TYPE_BRAIN_MODELS":
            cifti_vertices = cifti.header.get_axis(mi).vertex

    # TODO: Implement logging + proper error
    if cifti_vertices is None:
        raise ValueError("CIFTI object does not contain BrainModelAxis!")

    # Validate and obtain GIFTI
    for g in [l_gifti, r_gifti]:
        contains_pointset = any([
            True for d in g.darrays
            if d.intent == INTENT.NIFTI_INTENT_POINTSET.value
        ])
        if not contains_pointset:
            raise ValueError(f"{g.get_filename()} is not a surface mesh file!")

    # Extract vertices from GIFTI
    verts, trigs, _ = gifti_get_full_brain_mesh(l_gifti, r_gifti)

    # Map CIFTI vertices to GIFTI, setting non-filled values to NaN
    mapping_array = np.empty((cifti.dataobj.shape[0], verts.shape[0]),
                             dtype=cifti.dataobj.dtype)
    # Write NaNs
    mapping_array[:] = np.nan
    mapping_array[:, cifti_vertices] = cifti.get_fdata()

    # Return mapping array
    return verts, trigs, mapping_array
