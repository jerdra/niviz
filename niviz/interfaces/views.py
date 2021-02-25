# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from nilearn import plotting as nilp
import numpy as np
import svgutils.transform as svgt
import niworkflows.interfaces.report_base as nrc
from nipype.interfaces.base import File, traits, InputMultiPath
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

    seg_files = InputMultiPath(
                    File(exists=True,
                         usedefault=False,
                         resolve=True),
                    desc='Segmentation image of SVG',
                    mandatory=True)

    mask_file = File(exists=True,
                     resolve=True,
                     desc='ROI Mask for mosaic')
                     
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

    def _post_run_hook(self, runtime):
        '''
        Do nothing but propogate properties
        to (first) parent class of ISegRPT
        that is nrc.SegmentationRC
        '''

        # Set variables for `nrc.SegmentationRC`
        self._anat_file = self.inputs.anat_file
        self._seg_files = self.inputs.seg_files
        self._mask_file = self.inputs.mask_file or None
        self._masked = self.inputs.masked

        # Propogate to superclass
        return super(ISegRPT, self)._post_run_hook(runtime)

    def _run_interface(self, runtime):
        return runtime


class SurfaceRC(reporting.ReportCapableInterface):
    """
    Another abstract mixin for surface-type image QC reports
    
    This first version is based on nilearn's surface plotting capabilities.
    """     
    
    def _generate_report(self):
        """Generates a composite of surface parcellation results from the subjects's fsaverage dir"""
        
        ncols = len(self._views)
        svgs = []
        for view in self._views:
            svgs.append(svgt.from_mpl(
                nilp.plot_surf_roi(self._surf_mesh, 
                                   roi_map=self._roi_map,
                                   hemi=self._hemi, 
                                   view=view,
                                   bg_map=self._bg_map, 
                                   bg_on_data=True, 
                                   darkness=.5,
                                   title='%s - %s' % (self._hemi, view))))
        
        roots = [f.getroot() for f in svgs]
            
        dims = []
        for svg in svgs:
            viewbox = [float(v) for v in svg.root.get("viewBox").split(" ")]
            width = int(viewbox[2])
            height = int(viewbox[3])
            dims.append((width, height))
    
        
        dims = np.array(dims)
        width = dims[0,0]
        height = dims[0,1] 
        
        # Compose the svg roots into a single figure:
        fig = svgt.SVGFigure(width*ncols, height)
        
        offset = 0
        labels = []
        for i, r in enumerate(roots):
            r.moveto(offset, 0)
            label = '{} - {}'.format(self._hemi, self._views[i])
            labels.append(svgt.TextElement(offset, 
                                           round(height*.05), 
                                           label, 
                                           size=12, 
                                           weight='bold'))
            offset += width
        
        fig.append(roots) 
        fig.append(labels)
                        
        fig.save(self._out_report)    
        
        
class _ISurfInputSpecRPT(nrc._SVGReportCapableInputSpec):
    '''
    Input specification for ISurfRPT, implements:
    surf_mesh: surface mesh geometry, can be .gii or freesurfer specific such as .orig, .pial,
            .sphere, .white, .inflated)
    
    roi_map: ROI map to be displayed on the surface mesh, can be a file (valid formats are 
            .gii, .mgz, .nii, .nii.gz, or Freesurfer specific files such as .annot or .label), 
            or a Numpy array with a value for each vertex of the surf_mesh
            
    hemi: Hemisphere to display, default is 'left'
    
    bg_map: Background image to be plotted on the mesh underneath the stat_map in greyscale, 
            most likely a sulcal depth map for realistic shading.  
            
    views: Optional, a list of desired views to be included (e.g. lateral, medial, dorsal, ventral)
    
    Bases _SVGReportCapableInputSpec which implements:
    out_report: Filename trait
    compress_report: ["auto", true, false]
    '''
        
    surf_mesh = File(exists=True,
                     usedefault=False,
                     resolve=True,
                     desc='Surface mesh',
                     mandatory=True)

    roi_map = File(exists=True,
                 usedefault=False,
                 resolve=True,
                 desc='ROI map for surface mesh',
                 mandatory=True)

    hemi = traits.String(mandatory=False,
                usedefault=True,
                desc='left or right hemisphere')
                     
    bg_map = File(exists=True,
                  usedefault=False,
                  resolve=True,  
                  desc='Background image plotted on mesh',
                  mandatory=False)
    
    views = traits.List(traits.String(), 
                desc='List of views to include',
                mandatory=False)

class _ISurfOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class ISurfRPT(SurfaceRC):
    '''
    Class to generate surface meshes based on nilearn's plot_surf_roi 
    Supports various freesurfer outputs
    '''

    # Use our declared IO specs
    input_spec = _ISurfInputSpecRPT
    output_spec = _ISurfOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._surf_mesh = self.inputs.surf_mesh
        self._roi_map = self.inputs.roi_map
        self._hemi = self.inputs.hemi or 'left'
        self._views = self.inputs.views or ['lateral', 'medial']
        self._bg_map = self.inputs.bg_map or None

        return super(ISurfRPT, self)._post_run_hook(runtime)

    def _run_interface(self, runtime):
        return runtime
        
        
class _ISurfSegInputSpecRPT(nrc._SVGReportCapableInputSpec):
    '''
    Input specification for ISurfSegRPT, implements:
    anat_file: Input anatomical image
    mask_file: Input ROI mask
    contour_file: Input contour file (optional)
    
    Bases _SVGReportCapableInputSpec which implements:
    out_report: Filename trait
    compress_report: ["auto", true, false]
    '''
    anat_file = File(exists=True,
                     usedefault=False,
                     resolve=True,
                     desc='Anatomical image of SVG',
                     mandatory=True)

    contour_file = File(exists=True,
                       resolve=True,
                       desc='Contours to include in image',
                       mandatory=False)


class _ISurfSegOutputSpecRPT(reporting.ReportCapableOutputSpec):
    pass


class ISurfSegRPT(nrc.SurfaceSegmentationRC):
    '''
    Class to generate surface segmentation images from pre-existing .mgz files.
    '''

    # Use our declared IO specs
    input_spec = _ISurfSegInputSpecRPT
    output_spec = _ISurfSegOutputSpecRPT

    def _post_run_hook(self, runtime):
        '''
        Do nothing but propogate properties
        to (first) parent class of ISurfSegRPT
        that is nrc.SurfaceSegmentationRC
        '''
        outputs = self.aggregate_outputs(runtime=runtime)
        
        # Set variables for `nrc.SurfaceSegmentationRC`
        self._anat_file = self.inputs.anat_file
        self._masked = False
        self._contour = self.inputs.contour_file
        
        print('Generating report containing a surface segmentation (outputs: {})'.format(outputs))
        
        # Propogate to superclass
        return super(ISurfSegRPT, self)._post_run_hook(runtime)

    def _run_interface(self, runtime):
        return runtime
    