# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import niworkflows.interfaces.report_base as nrc
from nipype.interfaces.base import File
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
