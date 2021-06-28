"""
Provides utility functions to access Surface data
"""

import numpy as np

CIFTI_GIFTI_MAP = {
    'CIFTI_STRUCTURE_CORTEX_LEFT': 'left',
    'CIFTI_STRUCTURE_CORTEX_RIGHT': 'right',
    'CortexLeft': 'left',
    'CortexRight': 'right'
}

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


def map_cifti_to_gifti(gifti, cifti):
    '''
    Maps cifti data-array to gifti vertices to account
    for missing indices (i.e removal of medial wall)

    Arguments:
        gifti:      GIFTI surface mesh
        cifti:      CIFTI file to map [Series x BrainModel]

    Returns:
        verts:          Vertices of surface mesh
        trigs:          Triangles of surface mesh
        mapping_array:  An [Features x Vertices] mapping array pulled
                        from the CIFTI image
    '''

    # Validate and obtain CIFTI indices
    brain_models = None
    for mi in cifti.header.mapped_indices:
        map_type = cifti.header.get_index_map(mi).indices_map_to_data_type
        if map_type == "CIFTI_INDEX_TYPE_BRAIN_MODELS":
            brain_models = cifti.header.get_axis(mi)

    # TODO: Implement logging + proper error
    if brain_models is None:
        raise ValueError("CIFTI object does not contain BrainModelAxis!")

    # Validate and obtain GIFTI
    gifti_struct = None
    for d in gifti.darrays:
        if d.intent == NIFTI_INTENT_POINTSET:
            try:
                gifti_struct = d.metadata['AnatomicalStructurePrimary']
            except KeyError:
                raise ValueError(
                    f"{gifti.get_filename()} is not a valid surface mesh file!"
                )

    if gifti_struct is None:
        raise ValueError(
            f"{gifti.get_filename()} contains no coordinate information!")

    # Now we need to map the coordinate of CIFTI onto GIFTI
    match_key = CIFTI_GIFTI_MAP[gifti_struct]
    matched_bm = None
    for struct, sl, bma in brain_models.iter_structures():
        if CIFTI_GIFTI_MAP[struct] == match_key:
            matched_bm = (struct, sl, bma)
            break

    if matched_bm is None:
        raise ValueError(
            "No matching structures between CIFTI and GIFTI file!")

    _, matched_verts, brain_model_ax = matched_bm
    cifti_verts = brain_model_ax.vertex

    # Extract vertices from GIFTI
    verts, trigs = gifti_get_mesh(gifti)

    # Map CIFTI vertices to GIFTI, setting non-filled values to NaN
    mapping_array = np.empty((cifti.dataobj.shape[0], verts.shape[0]),
                             dtype=cifti.dataobj.dtype)

    # Write NaNs
    mapping_array[:] = np.nan
    try:
        mapping_array[:, cifti_verts] = cifti.get_fdata()[:, matched_verts]
    except IndexError:
        raise ValueError("Cifti file contains vertices that are not indexed "
                         "by the provided gifti file!")

    # Return mapping array
    return verts, trigs, mapping_array
