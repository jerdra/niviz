# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations
from typing import TYPE_CHECKING
import os
from collections import namedtuple

if TYPE_CHECKING:
    from nibabel.nifti1 import Nifti1Image

import niworkflows.interfaces.report_base as nrc
from nipype.interfaces.base import File, traits, InputMultiPath, Directory
from traits.trait_types import BaseInt
from nipype.interfaces.mixins import reporting
from niworkflows.viz.utils import cuts_from_bbox, compose_view

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nilearn.image
import nilearn.plotting as nplot
import nibabel as nib
import numpy as np

from ..node_factory import register_interface
import niviz.surface
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


class _FSInputSpecRPT(nrc._SVGReportCapableInputSpec):
    bg_nii = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc='Background NIFTI for SVG, will use T1.mgz if not '
                  'specified',
                  mandatory=False)

    fs_dir = Directory(exists=True,
                       usedefault=False,
                       resolve=True,
                       desc='Subject freesurfer directory',
                       mandatory=True)


class _IFSCoregInputSpecRPT(_FSInputSpecRPT):
    fg_nii = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc='Foreground NIFTI for SVG',
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


class _ParcellationInputSpecRPT(nrc._SVGReportCapableInputSpec):
    parcellation = File(exists=True,
                        usedefault=False,
                        resolve=True,
                        desc="Parcellated NIFTI file",
                        mandatory=True)
    colortable = File(exists=True,
                      usedefault=False,
                      resolve=True,
                      desc="Lookup color table for parcellation")


class _IParcellationOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class _IFreesurferVolParcellationInputSpecRPT(_ParcellationInputSpecRPT,
                                              _FSInputSpecRPT):
    pass


class _IFreesurferVolParcellationOutputSpecRPT(
        reporting.ReportCapableOutputSpec):
    pass


class _IFreeSurferVolParcellationRPT(reporting.ReportCapableInterface):
    '''
    Freesurfer-based Parcellation Report.

    Uses FreeSurferColorLUT table to map colors to integer values
    found in NIFTI file
    '''

    input_spec = _IFreesurferVolParcellationInputSpecRPT
    output_spec = _IFreesurferVolParcellationOutputSpecRPT

    def _post_run_hook(self, runtime: Bunch) -> Bunch:

        if not self.inputs.bg_nii:
            self._bg_nii = nib.load(
                os.path.join(self.inputs.fs_dir, "mri", "T1.mgz"))
        else:
            self._bg_nii = nib.load(self.inputs.bg_nii)

        # TODO: ENUM this to the available freesurfer parcellations
        parcellation = nib.load(self.inputs.parcellation)
        d_parcellation = parcellation.get_fdata(dtype=int)

        # Re-normalize the ROI values by rank
        # Then extract colors from full colortable using rank ordering
        unique_v, u_id = np.unique(d_parcellation.flatten())
        colormap = _parse_freesurfer_LUT(self.inputs.colortable)

        # Remap parcellation to rank ordering
        d_parcellation = u_id.reshape(d_parcellation)
        parcellation = nilearn.image.new_img_like(parcellation,
                                                  d_parcellation,
                                                  copy_header=True)

        # Resample to background resolution
        parcellation = nilearn.image.resample_to_img(parcellation,
                                                     self._bg_nii,
                                                     interpolation='nearest')
        self._segs = _parcel2segs(parcellation)

        # Get segmentation colors
        self._colors = [colormap[i] for i in unique_v]

        # Now we need to call the parent process
        return super(_IFreeSurferVolParcellationRPT,
                     self)._post_run_hook(runtime)

    def _run_interface(self, runtime: Bunch) -> Bunch:
        return runtime

    def _generate_report(self):
        '''
        Construct a parcellation overlay image
        '''
        from niworkflows.viz.utils import plot_segs

        compose_view(plot_segs(image_nii=self._bg_nii,
                               seg_niis=self._segs,
                               bbox_nii=None,
                               out_file=self.inputs.out_report,
                               colors=self._colors,
                               filled=True),
                     fg_svgs=None,
                     out_file=self._out_report)


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

        import trimesh

        l_surf = nib.load(self._surf_l)
        r_surf = nib.load(self._surf_r)
        vol_img = nib.load(self._bg_nii)

        if vol_img.ndim == 4:
            vol_img = vol_img.slicer[:, :, :, 0]

        verts, trigs, offset = niviz.surface.gifti_get_full_brain_mesh(
            l_surf, r_surf)

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


class _ISurfMapInputSpecRPT(nrc._SVGReportCapableInputSpec):

    left_surf = File(exists=True,
                     usedefault=False,
                     resolve=True,
                     desc="Left surface mesh",
                     mandatory=True)

    right_surf = File(exists=True,
                      usedefault=False,
                      resolve=True,
                      desc="Right surface mesh",
                      mandatory=True)

    bg_map = File(exists=True,
                  usedefault=False,
                  resolve=True,
                  desc="Cifti file containing background map data "
                  "(usually sulci depth)",
                  mandatory=False)

    cifti_map = File(exists=True,
                     usedefault=False,
                     resolve=True,
                     desc="Cifti file containing surface map data",
                     mandatory=False)

    colormap = traits.String("magma",
                             usedefault=True,
                             desc="Colormap to use to plot mapping",
                             mandatory=False)

    views = traits.List(["lateral", "medial"],
                        usedefault=True,
                        desc="Views to display",
                        inner_traits=traits.Enum(
                            values=["lateral", "medial", "dorsal", "ventral"]))
    darkness = traits.BaseInt(
        0.3,
        usedefault=True,
        desc="Multiplicative factor of bg_img onto foreground map",
        mandatory=False)
    visualize_all_maps = traits.Bool(False,
                                     usedefault=True,
                                     desc="Visualize all mappings in "
                                     "mapping file, if false will visualize "
                                     "only the first mapping")


class _ISurfMapOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class ISurfMapRPT(reporting.ReportCapableInterface):
    '''
    Class for generating Niviz surface visualizations given
    a mesh and surface mapping
    '''

    input_spec = _ISurfMapInputSpecRPT
    output_spec = _ISurfMapOutputSpecRPT

    def _run_interface(self, runtime: Bunch) -> Bunch:
        """Instantiation of abstract method, does nothing

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object (unchanged)

        """
        return runtime

    def _post_run_hook(self, runtime: Bunch) -> Bunch:
        self._left_surf = self.inputs.left_surf
        self._right_surf = self.inputs.right_surf
        self._cifti_map = self.inputs.cifti_map
        self._bg_map = self.inputs.bg_map
        self._views = self.inputs.views
        self._colormap = self.inputs.colormap
        self._visualize_all_maps = self.inputs.visualize_all_maps
        self._darkness = self.inputs.darkness

        return super(ISurfMapRPT, self)._post_run_hook(runtime)

    def _generate_report(self):
        """Side effect function of ISurfMapRPT

        Generate a surface visualization

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object
        """

        from mpl_toolkits import mplot3d  # noqa: F401

        Hemispheres = namedtuple("Hemispheres", ["left", "right"])

        l_surf = nib.load(self._left_surf)
        r_surf = nib.load(self._right_surf)
        num_views = len(self._views)
        num_maps = 1

        if self._cifti_map:
            cifti_map = nib.load(self._cifti_map)
            lv, lt, lm = niviz.surface.map_cifti_to_gifti(l_surf, cifti_map)
            rv, rt, rm = niviz.surface.map_cifti_to_gifti(r_surf, cifti_map)

            if not self._visualize_all_maps:
                lm = lm[0, :]
                rm = rm[0, :]
            else:
                num_maps = lm.shape[0]

            map_hemi = Hemispheres(left=(lv, lt, lm), right=(rv, rt, rm))
        else:
            # Use vertices and triangles from Mesh
            lv, lt = niviz.surface.gifti_get_mesh(l_surf)
            rv, rt = niviz.surface.gifti_get_mesh(r_surf)
            map_hemi = Hemispheres(left=(lv, lt, None), right=(lv, lt, None))

        if self._bg_map:
            bg_map = nib.load(self._bg_map)
            _, _, l_bg = niviz.surface.map_cifti_to_gifti(l_surf, bg_map)
            _, _, r_bg = niviz.surface.map_cifti_to_gifti(r_surf, bg_map)
            bg_hemi = Hemispheres(left=l_bg, right=r_bg)
        else:
            bg_hemi = Hemispheres(left=None, right=None)

        # Construct figure
        w, h = plt.figaspect(num_maps / (num_views * 2))
        fig, axs = plt.subplots(num_maps,
                                num_views * 2,
                                subplot_kw={'projection': '3d'},
                                figsize=(w, h))
        fig.set_facecolor("black")
        fig.tight_layout()
        for i, a in enumerate(axs.flat):
            a.set_facecolor("black")

            # Get row (map)
            i_map = i // num_views

            # Get column
            i_view = (i - i_map) % (num_views * 2)
            view = self._views[i_view // 2]

            # Get hemisphere
            hemi = i_view % 2
            if hemi == 0:
                display_map = map_hemi.left
                display_bg = bg_hemi.left
                hemi = "left"
            else:
                display_map = map_hemi.right
                display_bg = bg_hemi.right
                hemi = "right"

            # Plot
            v, t, m = display_map
            nplot.plot_surf([v, t],
                            surf_map=m,
                            bg_map=display_bg,
                            cmap=self._colormap,
                            axes=a,
                            hemi=hemi,
                            view=view,
                            bg_on_data=True,
                            darkness=self._darkness)

        plt.draw()
        plt.savefig(self._out_report)


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


def _parse_freesurfer_LUT(colortable: str) -> dict:
    '''
    Parse Freesurfer-style colortable into a
    matplotlib compatible categorical colormap

    Args:
        Path to Freesurfer colormap table

    Returns:
        Matplotlib colormap object encoding Freesurfer colors
    '''
    color_mapping = {}
    with open(colortable, 'w') as ct:
        for line in ct:
            if "#" in ct:
                continue
            roi, _, r, g, b, _ = [
                entry for entry in line.strip("\n").split(" ") if entry
            ]
            color_mapping[roi] = [r, g, b]

    return color_mapping


# TODO: Move plotting/helper utilities into own module
# https://stackoverflow.com/questions/1376438/how-to-make-a-repeating-generator-in-python
def multigen(gen_func):
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs

        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)

    return _multigen


@multigen
def _parcel2segs(parcellation):
    d_parcellation = parcellation.get_fdata(dtype=int)
    for i in np.unique(parcellation.get_fdata(dtype=int)):
        yield nilearn.image.new_img_like(parcellation, d_parcellation == i)


# Register interfaces with adapter-factory
def _run_imports() -> None:
    register_interface(IRegRPT, 'registration')
    register_interface(ISegRPT, 'segmentation')
    register_interface(IFSCoregRPT, 'freesurfer_coreg')
    register_interface(ISurfVolRPT, 'surface_coreg')
    register_interface(ISurfMapRPT, 'surface')
