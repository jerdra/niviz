import pytest
import numpy as np
import niviz.surface


@pytest.fixture
def cifti():
    import nibabel.cifti2.cifti2 as ncif
    import nibabel.cifti2.cifti2_axes as ncax

    np.random.seed(seed=1)
    verts = np.arange(0, 100, 2)
    data = np.random.uniform(size=(1, 50))

    ser_ax = ncax.ScalarAxis(name=["Data"])
    bm_ax = ncax.BrainModelAxis(name="CORTEX_LEFT",
                                vertex=verts,
                                affine=np.eye(4),
                                nvertices={"CORTEX_LEFT": verts.shape[0]})
    h = ncif.Cifti2Header.from_axes((ser_ax, bm_ax))
    c = ncif.Cifti2Image(data, h)
    return c


@pytest.fixture
def bad_cifti():
    import nibabel.cifti2.cifti2 as ncif
    import nibabel.cifti2.cifti2_axes as ncax

    np.random.seed(seed=1)
    data = np.random.uniform(size=(1, 50))

    ser_ax = ncax.ScalarAxis(name=["Data"])
    h = ncif.Cifti2Header.from_axes((ser_ax, ser_ax))
    c = ncif.Cifti2Image(data, h)
    return c


@pytest.fixture
def gifti():

    import nibabel.gifti as ngif

    g = ngif.GiftiImage()
    coords = np.random.uniform(size=(100, 3))
    trigs = np.random.randint(low=0, high=100, size=(5, 3))

    struct = ngif.GiftiNVPairs(name="AnatomicalStructurePrimary",
                               value="CortexLeft")
    meta = ngif.GiftiMetaData(struct)
    darray_coords = ngif.GiftiDataArray(data=coords,
                                        intent='NIFTI_INTENT_POINTSET',
                                        meta=meta)
    darray_trigs = ngif.GiftiDataArray(data=trigs, intent=1009)
    g.add_gifti_data_array(darray_coords)
    g.add_gifti_data_array(darray_trigs)
    return g


@pytest.fixture
def bad_gifti():

    import nibabel.gifti as ngif

    g = ngif.GiftiImage()
    coords = np.random.uniform(size=(20, 3))
    trigs = np.random.randint(low=0, high=100, size=(5, 3))

    struct = ngif.GiftiNVPairs(name="AnatomicalStructurePrimary",
                               value="CortexLeft")
    meta = ngif.GiftiMetaData(struct)
    darray_coords = ngif.GiftiDataArray(data=coords,
                                        intent='NIFTI_INTENT_POINTSET',
                                        meta=meta)
    darray_trigs = ngif.GiftiDataArray(data=trigs, intent=1009)
    g.add_gifti_data_array(darray_coords)
    g.add_gifti_data_array(darray_trigs)
    return g


def test_map_cifti2gifti_maps_correct_vertices(cifti, gifti):

    import numpy as np

    _, _, m = niviz.surface.map_cifti_to_gifti(gifti, cifti)

    assert any(np.isnan(m[0, ::2])) is False
    assert all(np.isnan(m[0, 1::2]))


def test_map_cifti2gifti_fails_when_not_matched_surface(cifti, gifti):

    import nibabel.gifti as ngif
    badNV = ngif.GiftiNVPairs(name="BAD", value="BAD")
    gifti.darrays[0].meta = ngif.GiftiMetaData(badNV)

    with pytest.raises(ValueError):
        niviz.surface.map_cifti_to_gifti(gifti, cifti)


def test_map_cifti2gifti_fails_when_no_brain_model_in_cifti(bad_cifti, gifti):

    with pytest.raises(ValueError):
        niviz.surface.map_cifti_to_gifti(gifti, bad_cifti)


def test_map_cifti2gifti_fails_when_mismatch_in_vertices(cifti, bad_gifti):

    with pytest.raises(ValueError):
        niviz.surface.map_cifti_to_gifti(bad_gifti, cifti)
