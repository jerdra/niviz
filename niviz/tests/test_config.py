import niviz.config
from distutils import dir_util
import pytest
import os
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


@pytest.fixture
def datadir(tmpdir, request):
    '''
    Pull fixture from directory with the same file name as module
    https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    '''

    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


def test_config_fails_when_environment_variable_missing(datadir):
    '''
    Test whether when environment variables are defined in the input
    specification but not in the calling environment that a ValidationError
    will be raised
    '''

    spec = datadir.join("only-environment-fixture.yml")
    with pytest.raises(niviz.config.ValidationError):
        niviz.config.SpecConfig(spec, '')


def test_config_substitutes_env_when_variable_available(datadir):
    '''
    Test whether environment variables are correctly substituted into
    the YAML file when available
    '''

    # Set environment variable
    os.environ["NOTDEFINED"] = "DEFINED"
    spec = datadir.join("only-environment-fixture.yml")
    config = niviz.config.SpecConfig(spec, '')
    expected_defaults = {"NOTDEFINED": "DEFINED"}
    assert config.defaults["env"] == expected_defaults


def test_update_specs_with_default_updates_correctly(datadir):
    '''
    Test whether an image generation specification correctly
    inherits settings specified as globals
    '''

    # Construct config object
    input_spec = datadir.join("file-spec-fixture.yml")
    config = niviz.config.SpecConfig(input_spec, '')

    # Load fixture directly
    with open(input_spec, 'r') as f:
        raw_spec = yaml.load(f, Loader=Loader)['filespecs'][0]

    expected_bidsmap = {
        "sub": {
            "value": "subject_value"
        },
        "desc": {
            "value": "desc"
        }
    }

    expected_bids_hierarchy = ["sub", "ses"]

    expected_args = [{
        "field": "bg_nii",
        "value": "test",
        "path": True
    }, {
        "field": "fg_nii",
        "value": "SOMEVAR",
        "path": True
    }]

    res = config._update_spec_with_defaults(raw_spec)

    assert res["bids_map"] == expected_bidsmap
    assert res["bids_hierarchy"] == expected_bids_hierarchy
    assert res["args"] == expected_args


def test_bids_entities_are_correctly_extracted_from_path():
    '''
    Test whether given a bids_map and path to parse that
    BIDS entities are correctly pulled
    '''
    spec = {
        "bids_map": {
            "sub": {
                "value": "(?<=sub-)[A-Za-z0-9]+",
                "regex": True
            },
            "acq": {
                "value": "(?<=acq-)[A-Za-z0-9]+",
                "regex": True
            }
        }
    }

    file_spec = niviz.config.FileSpec(spec)
    expected_entities = {"sub": "ABCD", "acq": None}
    path = "/this/is/a/fake/path/sub-ABCD_fake-ignore"

    res = file_spec._extract_bids_entities(path)
    assert expected_entities == res


def test_bids_entities_fails_when_no_entities_are_found():
    '''
    If entities are expected but none are found then should fail
    '''
    spec = {
        "bids_map": {
            "sub": {
                "value": "(?<=sub-)[A-Za-z0-9]+",
                "regex": True
            },
            "acq": {
                "value": "(?<=acq-)[A-Za-z0-9]+",
                "regex": True
            }
        }
    }

    file_spec = niviz.config.FileSpec(spec)
    bad_path = "thisisbad"
    with pytest.raises(ValueError):
        file_spec._extract_bids_entities(bad_path)


def test_matching_algorithm_correctly_spreads_entities():
    '''
    Test whether less specific BIDS entities are spread to more
    specific BIDS entities with the matching algorithm
    '''

    input_mapping = (({
        "sub": "A",
        "ses": "B",
        "task": "rest"
    }, "a"), ({
        "sub": "A",
        "ses": "B",
        "task": "faces"
    }, "b"), ({
        "sub": "A",
        "ses": "B",
        "task": None
    }, "c"))
    bids_hierarchy = ["sub", "ses", "task"]

    expected_groups = {
        (("sub", "A"), ("ses", "B"), ("task", "rest")): ["a", "c"],
        (("sub", "A"), ("ses", "B"), ("task", "faces")): ["b", "c"]
    }

    file_spec = niviz.config.FileSpec({"bids_hierarchy": bids_hierarchy})
    res = file_spec._group_by_hierarchy(input_mapping, bids_hierarchy)

    # Now check that each group is matched
    for k, v in expected_groups.items():
        assert res[k] == v


def test_gen_args_makes_correct_output_when_cropped_hierarchy():
    '''
    Ensure that when an entity is not specified to be matched on in
    the hierarchy that we don't use it as a matching criteria
    and instead are grouped
    '''
    input_mapping = (({
        "sub": "A",
        "ses": "B",
        "task": "rest"
    }, "a"), ({
        "sub": "A",
        "ses": "B",
        "task": "faces"
    }, "b"), ({
        "sub": "A",
        "ses": "B",
        "task": None
    }, "c"))
    bids_hierarchy = ["sub", "ses"]

    expected_groups = {
        (("sub", "A"), ("ses", "B")): ["a", "b", "c"],
    }

    file_spec = niviz.config.FileSpec({"bids_hierarchy": bids_hierarchy})
    res = file_spec._group_by_hierarchy(input_mapping, bids_hierarchy)

    # Now check that each group is matched
    for k, v in expected_groups.items():
        assert res[k] == v


def test_end_to_end_filespec_generation(datadir):
    filespec = datadir.join("sample-data-spec.yml")
    basepath = datadir.join("sample-data")

    expected_names = ["test"] * 3
    expected_interface_args = [{
        "pathfield": f"{basepath}/sub-A/ses-01/sub-A_ses-01_task-aa_leaf",
        "spreadfield": f"{basepath}/sub-A/sub-A_spread"
    }, {
        "pathfield": f"{basepath}/sub-A/ses-01/sub-A_ses-01_task-bb_leaf",
        "spreadfield": f"{basepath}/sub-A/sub-A_spread"
    }, {
        "pathfield": f"{basepath}/sub-B/ses-01/sub-B_ses-01_task-aa_leaf",
        "spreadfield": f"{basepath}/sub-B/sub-B_spread"
    }]
    expected_methods = ["testmethod"] * 3
    expected_out_spec = [
        "sub-A_task-aa_desc-SOMEVAR.png",
        "sub-A_task-bb_desc-SOMEVAR.png",
        "sub-B_task-aa_desc-SOMEVAR.png",
    ]

    res = sorted(niviz.config.fetch_data(filespec, basepath),
                 key=lambda x: x._out_spec)
    expected_zip = zip(expected_names, expected_interface_args,
                       expected_methods, expected_out_spec)

    for i, namo in enumerate(expected_zip):
        name, arg, method, outspec = namo
        assert res[i].name == name
        assert res[i].interface_args == arg
        assert res[i].method == method
        assert res[i]._out_spec.name == outspec
