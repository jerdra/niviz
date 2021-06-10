import niviz.config
from distutils import dir_util
from pytest import fixture
import os
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

## Fixture
@fixture
def datadir(tmpdir, request):
    '''
    Pull fixture from directory with the same file name as module
    https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    '''

    filename = request.module.__file__
    test_dir, _ = os.path.splittext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, bytes(trmpdir))

    return tmpdir


def test_config_fails_when_environment_variable_missing(datadir):
    '''
    Test whether when environment variables are defined in the input
    specification but not in the calling environment that a ValidationError
    will be raised
    '''

    spec = yaml.load(datadir.join("only-environment-fixture.yml"))
    with pytest.raises(niviz.config.ValidationError):
        niviz.config.SpecConfig(spec, '')

def test_config_substitutes_env_when_variable_available(datadir):
    '''
    Test whether environment variables are correctly substituted into
    the YAML file when available
    '''

    # Set environment variable
    os.environ["NOTDEFINED"] = "DEFINED"
    spec = yaml.load(datadir.join("only-environment-fixture.yml"))
    config = niviz.config.SpecConfig(default, '')
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
    raw_spec = yaml.load(input_spec, Loader=Loader)['filespec'][0]


    expected_bidsmap = {
            "bids_map": {
                "sub": { "value": "subject_value" },
                "desc": { "value": "SOMEVAR" }
            }
    }

    expected_bids_hierarchy = ["sub", "ses"]

    expected_args = [
            {"field": "bg_nii", "value": "test", "path": True},
            {"field": "fg_nii", "value": "SOMEVAR", "path": True}
    ]

    res = config._update_spec_with_defaults(raw_spec)

    assert res["bids_map"] == expected_bidsmap
    assert res["bids_hierarchy"] == expected_bids_hierarchy
    assert res["args"] == expected_args

def test_bids_entities_are_correctly_extracted_from_path():
    '''
    Test whether given a bids_map and path to parse that
    BIDS entities are correctly pulled
    '''
    spec = { "bids_map": {
            "sub": { "value": "(?<=sub-)[A-Za-z0-9]+", "regex": True },
            "desc": { "value": "constantvalue"},
            "acq": { "value": "(?<=acq-)[A-Za-z0-9]+", "regex": True }
        }}

    file_spec = niviz.config.FileSpec(spec)
    expected_entities = {"sub": "ABCD", "desc": "constantvalue",
                        "acq": None}
    path = "/this/is/a/fake/path/sub-ABCD_desc-constantvalue_fake-BAD"

    res = file_spec._extract_bids_entities(path)
    assert expected_entities = res


def test_bids_entities_fails_when_no_entities_are_found():
    '''
    If entities are expected but none are found then should fail
    '''
    spec = { "bids_map": {
            "sub": { "value": "(?<=sub-)[A-Za-z0-9]+", "regex": True },
            "desc": { "value": "constantvalue"},
            "acq": { "value": "(?<=acq-)[A-Za-z0-9]+", "regex": True }
        }}

    file_spec = niviz.config.FileSpec(spec)
    bad_path = "thisisbad"
    with pytest.raises(TypeError):
        file_spec._extract_bids_entities(bad_path)

def test_matching_algorithm_correctly_groups_data():
    '''
    Test whether less specific BIDS entities are spread to more
    specific BIDS entities with the matching algorithm

    In addition the most specified entities should have their
    BIDS entities matched exactly
    '''
    pass
    
