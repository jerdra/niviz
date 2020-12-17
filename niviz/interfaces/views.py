# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import niworkflows.interfaces.report_base as nrc
from nipype.interfaces.base import File
from nipype.interfaces.base import InputMultiPath
from nipype.interfaces.mixins import reporting

"""
ReportCapable concrete classes for generating reports as side-effects
"""


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
                    mandatory=True)


class _IRegOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class IRegRPT(nrc.RegistrationRC):

    input_spec = _IRegInputSpecRPT
    output_spec = _IRegOutputSpecRPT

    def _post_run_hook(self, runtime):

        self._fixed_image = self.inputs.bg_nii
        self._moving_image = self.inputs.fg_nii

        return super(IRegRPT, self)._post_run_hook(runtime)

    def _run_interface(self, runtime):
        return runtime


class _ISegInputSpecRPT(nrc._SVGReportCapableInputSpec,
                        BaseInterfaceInputSpec):
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

    seg_files = InputMultiPath(
                    File(exists=True,
                         usedefault=False,
                         resolve=True),
                    desc='Segmentation image of SVG',
                    mandatory=True)

    mask_file = File(exists=True,
                     udedefault=False,
                     resolve=True,
                     desc='ROI Mask for mosaic',
                     mandatory=True)


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

    def _post_run_hook(self, runtime):
        '''
        Do nothing but propogate properties
        to (first) parent class of ISegRPT
        that is nrc.SegmentationRC
        '''

        # Set variables for `nrc.SegmentationRC`
        self._anat_file = self.inputs.anat_file
        self._seg_files = self.inputs.seg_files
        self._mask_file = self.inputs.mask_file
        self._masked = True

        # Propogate to superclass
        return super(ISegRPT, self)._post_run_hook(runtime)

    def _run_interface(self, runtime):
        return runtime
