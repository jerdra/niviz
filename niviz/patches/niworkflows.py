'''
Monkey patches to make niworkflows just a tad bit more flexible
Forgive me for I hath sinned
'''
import nibabel as nib
import nilearn.image as nimg
from nilearn.plotting import plot_anat


def _3d_in_file(in_file):
    """ if self.inputs.in_file is 3d, return it.
    if 4d, pick an arbitrary volume and return that.

    if in_file is a list of files, return an arbitrary file from
    the list, and an arbitrary volume from that file
    """
    from nipype.utils import filemanip

    if not isinstance(in_file, nib.filebasedimages.SerializableImage):
        in_file = filemanip.filename_to_list(in_file)[0]

    try:
        in_file = nib.load(in_file)
    except AttributeError:
        in_file = in_file
    except TypeError:
        in_file = in_file

    if len(in_file.shape) == 3:
        return in_file

    return nimg.index_img(in_file, 0)


def _plot_anat_with_contours(image, segs=None, compress="auto", **plot_params):
    '''
    Patched version of _plot_anat_with_contours that will
    accept
    '''

    from niworkflows.viz.utils import extract_svg
    nsegs = len(segs or [])
    plot_params = plot_params or {}
    # plot_params' values can be None, however they MUST NOT
    # be None for colors and levels from this point on.
    colors = plot_params.pop("colors", None) or []
    levels = plot_params.pop("levels", None) or []
    filled = plot_params.pop("filled", False)
    alpha = plot_params.pop("alpha", 1)
    missing = nsegs - len(colors)
    if missing > 0:  # missing may be negative
        from seaborn import color_palette

        colors = colors + color_palette("husl", missing)

    colors = [[c] if not isinstance(c, list) else c for c in colors]

    # Handle when a sequence of RGB is given as colors
    if not isinstance(colors[0], str):
        from matplotlib.colors import to_hex
        colors = [to_hex(c) for c in colors]

    if not levels:
        levels = [[0.5]] * nsegs

    # anatomical
    # this line is problematic
    display = plot_anat(image, **plot_params)

    # remove plot_anat -specific parameters
    plot_params.pop("display_mode")
    plot_params.pop("cut_coords")

    plot_params["linewidths"] = 0.5
    for i in reversed(range(nsegs)):
        plot_params["colors"] = colors[i]
        display.add_contours(segs[i],
                             levels=levels[i],
                             filled=filled,
                             alpha=alpha,
                             **plot_params)

    svg = extract_svg(display, compress=compress)
    display.close()
    return svg
