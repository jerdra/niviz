# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations
from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    from nibael.nifti1 import Nifti1Image

import niworkflows.interfaces.report_base as nrc
from niworkflows.viz.utils import cuts_from_bbox
from nipype.interfaces.base import File, traits, InputMultiPath, Directory
from traits.trait_types import BaseInt
from nipype.interfaces.mixins import reporting

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nilearn.image
import nilearn.plotting as nplot
import nibabel as nib
import numpy as np
import trimesh

from ..node_factory import register_interface
"""
ReportCapable concrete classes for generating reports as side-effects
"""

if TYPE_CHECKING:
    from nipype.interfaces.base.support import Bunch


class _IRegInputSpecRPT(nrc._SVGReportCapableInputSpec):

    bg_nii = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc='Background NIFTI for SVG',
                  mandatory=True)

    fg_nii = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc='Foreground NIFTI for SVG',
                  mandatory=True)

    contours = File(exists=True,
                    usedefault=False,
                    resolve=True,
                    desc='Contours to include in image',
                    mandatory=False)


class _IRegOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class IRegRPT(nrc.RegistrationRC):
    """Implementation of Identity operation on RegistrationRC

    This class performs no operations and generates a report
    as a side-effect. It is primarily used to generate registration
    reports on already registered data.

    """

    input_spec = _IRegInputSpecRPT
    output_spec = _IRegOutputSpecRPT

    def _post_run_hook(self, runtime: Bunch) -> Bunch:
        """Side-effect function of IRegRPT.

        Generate transition report as a side-effect. No operations
        are performed on the data (identity)

        If a 4D image is passed in the first index will be pulled for viewing

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object propogated through ReportCapable
            interfaces

        """

        # Need to 3Dify 4D images and re-orient to RAS
        fi = _make_3d_from_4d(nilearn.image.load_img(self.inputs.fg_nii))
        bi = _make_3d_from_4d(nilearn.image.load_img(self.inputs.bg_nii))
        self._fixed_image = fi
        self._moving_image = bi

        return super(IRegRPT, self)._post_run_hook(runtime)

    def _run_interface(self, runtime: Bunch) -> Bunch:
        """Main function of IRegRPT, does nothing.

        Implements identity operation. IRegRPT expects
        fully registered inputs, so no operations are performed.

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object (unchanged)

        """
        return runtime


class _ISegInputSpecRPT(nrc._SVGReportCapableInputSpec):
    '''
    Input specification for ISegRPT, implements:

    anat_file: Input anatomical image
    seg_files: Input segmentation image(s) - can be a list or a single file
    mask_file: Input ROI mask

    Bases _SVGReportCapableInputSpec which implements:

    out_report: Filename trait
    compress_report: ["auto", true, false]

    '''
    anat_file = File(exists=True,
                     usedefault=False,
                     resolve=True,
                     desc='Anatomical image of SVG',
                     mandatory=True)

    seg_files = InputMultiPath(File(exists=True,
                                    usedefault=False,
                                    resolve=True),
                               desc='Segmentation image of SVG',
                               mandatory=True)

    mask_file = File(exists=True,
                     resolve=True,
                     desc='ROI Mask for mosaic',
                     mandatory=False)

    masked = traits.Bool(False,
                         usedefault=True,
                         desc='Flag to indicate whether'
                         ' image is already masked')


class _ISegOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class ISegRPT(nrc.SegmentationRC):
    '''
    Class to generate registration images from pre-existing
    NIFTI files.

    Effectively acts as an Identity node with report
    generation as a side-effect.
    '''

    # Use our declared IO specs
    input_spec = _ISegInputSpecRPT
    output_spec = _ISegOutputSpecRPT

    def _post_run_hook(self, runtime: Bunch) -> Bunch:
        """Side-effect function of ISegRPT.

        Generate transition report as a side-effect. No operations
        are performed on the data (identity)

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object propogated through ReportCapable
            interfaces

        """

        if not isinstance(self.inputs.seg_files, list):
            self.inputs.seg_files = [self.inputs.seg_files]

        # Set variables for `nrc.SegmentationRC`
        self._anat_file = self.inputs.anat_file
        self._seg_files = self.inputs.seg_files
        self._mask_file = self.inputs.mask_file or None
        self._masked = self.inputs.masked

        # Propogate to superclass
        return super(ISegRPT, self)._post_run_hook(runtime)

    def _run_interface(self, runtime: Bunch) -> Bunch:
        """Main function of ISegRPT, does nothing.

        Implements identity operation. ISegRPT expects
        fully registered inputs, so no operations are performed.

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object (unchanged)

        """
        return runtime


class _IFSCoregInputSpecRPT(nrc._SVGReportCapableInputSpec):

    bg_nii = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc='Background NIFTI for SVG',
                  mandatory=True)

    fg_nii = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc='Foreground NIFTI for SVG',
                  mandatory=True)

    fs_dir = Directory(exists=True,
                       usedefault=False,
                       resolve=True,
                       desc='Subject freesurfer directory',
                       mandatory=True)


class _IFSCoregOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class IFSCoregRPT(nrc.RegistrationRC):

    input_spec = _IFSCoregInputSpecRPT
    output_spec = _IFSCoregOutputSpecRPT

    def _post_run_hook(self, runtime: Bunch) -> Bunch:
        """Side-effect function of IFSCoregRPT.

        Generates Freesurfer-based EPI2T1 coregistration report
        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object propogated through ReportCapable
            interfaces

        """

        self._fixed_image = self.inputs.bg_nii
        self._moving_image = self.inputs.fg_nii
        self._contour = os.path.join(self.inputs.fs_dir, 'mri', 'ribbon.mgz')

        return super(IFSCoregRPT, self)._post_run_hook(runtime)

    def _run_interface(self, runtime: Bunch) -> Bunch:
        """Does nothing.

        Implements identity operation. IFSCoregRPT expects
        fully registered inputs, so no operations are performed.

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object (unchanged)

        """
        return runtime


def _make_3d_from_4d(nii: Nifti1Image, ind: int = 0) -> Nifti1Image:
    '''
    Convert 4D Image into 3D one by pulling a single volume.
    Performs identity mapping if input image is 3D

    Args:
        nii: Input image
        ind: Index to pull from 4D image
    '''

    if len(nii.shape) < 4:
        return nii

    return nii.slicer[:, :, :, ind]


def _reorient_to_ras(img: Nifti1Image) -> Nifti1Image:
    '''
    Re-orient image to RAS

    Args:
        img: Image to re-orient to match ref image

    Returns:
        img re-oriented to RAS
    '''

    img = nilearn.image.load_img(img)
    ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    img_ornt = nib.orientations.axcodes2ornt(
        nib.orientations.aff2axcodes(img.affine))
    img2ref = nib.orientations.ornt_transform(img_ornt, ras_ornt)
    return img.as_reoriented(img2ref)


# Register interfaces with adapter-factory
def _run_imports() -> None:
    register_interface(IRegRPT, 'registration')
    register_interface(ISegRPT, 'segmentation')
    register_interface(IFSCoregRPT, 'freesurfer_coreg')
    register_interface(ISurfVolRPT, 'surface_coreg')


class _ISurfVolInputSpecRPT(nrc._SVGReportCapableInputSpec):
    '''
    Input spec for reports coregistering surface and volume images

    '''
    bg_nii = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc='Background NIFTI for SVG',
                  mandatory=True)

    fg_nii = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc='Foreground NIFTI for SVG')

    surf_l = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc='Left surface file',
                  mandatory=True)

    surf_r = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc='Right surface file',
                  mandatory=True)

    n_cuts = BaseInt(desc='Number of slices to display')


class _ISurfVolOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


def gifti_get_mesh(gifti):
    '''
    Extract vertices and triangles from GIFTI surf.gii
    file
    
    Arguments:
        gifti (GiftiImage): Input GiftiImage
    '''

    v, t = gifti.agg_data(('pointset', 'triangle'))
    return v.copy(), t.copy()


def gifti_get_full_brain(l, r):
    '''
    Construct a full brain mesh by joining
    both hemispheres
    
    Arguments:
        l: Left hemisphere GiftiImage
        r: Right hemisphere GiftiImage
    '''
    l_vert, l_trig = gifti_get_mesh(l)
    r_vert, r_trig = gifti_get_mesh(r)

    offset = l_trig.max() + 1
    r_trig += offset

    verts = np.vstack((l_vert, r_vert))
    trigs = np.vstack((l_trig, r_trig))

    return (verts, trigs, offset)


class SurfVolRC(reporting.ReportCapableInterface):
    '''
    Abstract mixin for surface-volume coregistered images
    '''
    pass


class ISurfVolRPT(SurfVolRC):
    '''
    Report interface for co-registered surface/volumetric images 
    '''
    input_spec = _ISurfVolInputSpecRPT
    output_spec = _ISurfVolOutputSpecRPT

    def _post_run_hook(self, runtime):
        outputs = self.aggregate_outputs(runtime=runtime)

        self._bg_nii = self.inputs.bg_nii
        self._fg_nii = self.inputs.fg_nii or None
        self._surf_l = self.inputs.surf_l
        self._surf_r = self.inputs.surf_r
        self._ncuts = self.inputs.n_cuts or 7

        # Propogate to superclass
        return super(ISurfVolRPT, self)._post_run_hook(runtime)

    def _run_interface(self, runtime):
        return runtime

    def _generate_report(self):
        '''Make a composite for co-registration of surface and volume images'''

        l_surf = nib.load(self._surf_l)
        r_surf = nib.load(self._surf_r)
        vol_img = nib.load(self._bg_nii)

        if vol_img.ndim == 4:
            vol_img = vol_img.slicer[:, :, :, 0]

        verts, trigs, offset = gifti_get_full_brain(l_surf, r_surf)

        mesh = trimesh.Trimesh(vertices=verts, faces=trigs)
        mask_nii = nilearn.image.threshold_img(vol_img, 1e-3)
        cuts = cuts_from_bbox(mask_nii, cuts=self._ncuts)

        sections = mesh.section_multiplane(plane_normal=[0, 0, 1],
                                           plane_origin=[0, 0, 0],
                                           heights=cuts['z'])

        zh = nplot.plot_anat(vol_img, display_mode='z', cut_coords=cuts['z'])

        for z, s in zip(cuts['z'], sections):
            ax = zh.axes[z].ax
            if s:
                for segs in s.discrete:
                    ax.plot(*segs.T, color='r', linewidth=0.5)

        if self._fg_nii:
            fg_img = nib.load(self._fg_nii).slicer[:, :, :, 0]
            fg_img = nilearn.image.resample_to_img(fg_img,
                                                   vol_img,
                                                   interpolation="linear")
            # Custom colormap with transparencies
            ncolors = 256
            basecmap = 'viridis_r'
            color_array = plt.get_cmap(basecmap)(range(ncolors))
            color_array[:, -1] = np.linspace(1.0, 0.0, ncolors)

            # Set background intensity=0 to transparent
            color_array[0, :] = 0
            cmapviridis = mcolors.LinearSegmentedColormap.from_list(
                basecmap, colors=color_array)

            zh.add_overlay(fg_img, cmap=cmapviridis)

        zh.savefig(self._out_report)
