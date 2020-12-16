# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations
from typing import TYPE_CHECKING

import niworkflows.interfaces.report_base as nrc
from nipype.interfaces.base import File
from nipype.interfaces.mixins import reporting
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
                    mandatory=True)


class _IRegOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class IRegRPT(nrc.RegistrationRC):
    """Implementation of Identity operation on RegistrationRC

    This class performs no operations and generates a report
    as a side-effect. It is primarily used to generate registration
    reports on already registered data.

    Attributes:
        input_spec: Input specification holding (bg_nii, fg_nii, contours)
        output_spec: Contains output_report path
    """

    input_spec = _IRegInputSpecRPT
    output_spec = _IRegOutputSpecRPT

    def _post_run_hook(self, runtime: Bunch) -> Bunch:
        """Side-effect function of IRegRPT

        Generate transition report as a side-effect. No operations
        are performed on the data (identity)

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object propogated through ReportCapable
            interfaces

        """

        self._fixed_image = self.inputs.bg_nii
        self._moving_image = self.inputs.fg_nii

        return super(IRegRPT, self)._post_run_hook(runtime)

    def _run_interface(self, runtime: Bunch) -> Bunch:
        """Main function of IRegRPT, does nothing

        Implements identity operation. IRegRPT expects
        fully registered inputs, so no operations are performed.

        Args:
            runtime: Nipype runtime object

        Returns:
            runtime: Resultant runtime object (unchanged)

        """
        return runtime
