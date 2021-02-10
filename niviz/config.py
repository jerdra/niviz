import os
import logging
import logging.config
import collections.abc
from string import Template

import yaml
import re
import glob.glob as glob
from itertools import groupby
from dataclasses import dataclass

logging.config.fileConfig("logging.conf")

# Initialize module logger
logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when Configuration File is incorrectly specified"""
    pass


def _nested_update(d, u):
    '''
    Recursive updating of nested dict
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    '''
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _nested_update(d.get(k, {}), v)
        else:
            d[k] = v


def _prefix_path(x, prefix):
    '''
    Prefix path with root directory
    '''
    if x.startswith("."):
        return os.path.join(prefix, x[1:])
    else:
        return x


class SpecConfig(object):
    '''
    Class to provide interface to configuration
    specs for sourcing QC input files
    '''
    def __init__(self, yml, schema):

        # Validate yaml object and store original file
        config = yaml.load(yml)
        self._validate(yml, schema)
        self._yaml = yml

        defaults = config.get("global", {})

        # TODO: Remove when validation is implemented
        try:
            self.file_specs = config["filespecs"]
        except KeyError:
            logger.error("Missing filespecs list in YAML file!")
            raise ValidationError

        if 'env' in defaults:
            defaults['env'] = {
                k: self._substitute_env(v)
                for k, v in defaults['env'].iteritems()
            }

        self.defaults = defaults

    def _substitute_env(self, env):

        r = os.path.expandvars(env)
        unresolved = re.findall("\\$[A-Za-z0-9]", r)

        if unresolved:
            [
                logger.error(f"Undefined environment variable {u}!")
                for u in unresolved
            ]
            raise ValidationError
        return r

    def _validate(self, yml, schema):
        '''
        Validate YAML file

        Raises:
            ValidationError
        '''

        pass

    def get_file_args(self, base_path):
        '''
        Construct arg list
        '''

        return [self._get_file_arg(f, base_path) for f in self.file_specs]

    def _get_file_arg(self, spec, base_path):
        '''
        Construct arg
        '''

        _spec = _nested_update(spec, self.defaults.get('bids_map', {}))
        _spec['args'] = self._apply_envs(spec['args'])
        return FileSpec(_spec).gen_args(base_path)

    def _apply_envs(self, args):

        if 'env' not in self.defaults:
            return args

        # For each environment variable
        arg_list = []
        for f in args:
            f['value'] = Template(args['value']).substitute(
                self.defaults['env'])
            arg_list.append(f)

        return arg_list


class FileSpec(object):
    '''
    Class to implement QcSpec
    '''
    def __init__(self, spec, base_path=None):

        self.spec = spec

        if base_path:
            self.spec = {
                f: _prefix_path(v, base_path)
                for f, v in self.iter_args()
            }

    @property
    def name(self):
        return self.spec['name']

    @property
    def method(self):
        return self.spec['method']

    @property
    def args(self):
        return self.spec['args']

    def iter_args(self):
        for f in self.args:

            bids = f['nobids'] if 'nobids' in f else False
            yield (f['field'], f['value'], bids)

    @property
    def bids_map(self):
        return self.spec['bids_map']

    @property
    def out_path(self):
        return self.spec['out_path']

    @property
    def match_on(self):
        return self.spec['match_on']

    def _extract_bids_entities(self, path):
        '''
        Extract BIDS entities from path

        Raises ValueError if all keys in bids_map cannot
        be found for a given path

        Returns a tuple of BIDS (field,value) pairs
        '''

        res = []
        raise_error = False
        for k, v in self.bids_map.iteritems():

            if 'regex' in v.keys():
                try:
                    bids_val = re.search(v['value'], path)[0]
                except IndexError:
                    logger.error(
                        f"Cannot extract {k} from {path} using {v['regex']}!")
                    raise_error = True
            else:
                bids_val = v['value']

            res.append((k, bids_val))

        if raise_error:
            logger.error("Was not able to extract some BIDS fields, "
                         "some paths are missing BIDS information!")
            raise ValueError

        return tuple(res)

    def gen_args(self, base_path):
        # TODO: Add docstring

        # TODO: Consider making args a class or dataclass
        bids_results = []
        static_results = []
        for f, v, nobids in self.iter_args():
            for p in glob(f"{v}", recursive=True):

                cur_mapping = ({
                    "field": f,
                    "path": p,
                })

                if nobids:
                    static_results.append(cur_mapping)
                else:
                    bids_entities = self._extract_bids_entities(p)
                    bids_results.append((bids_entities, cur_mapping))

        matched = groupby(bids_results, lambda x: x['bids'])

        arg_specs = []
        for bids_entities, grouped in matched:

            bids_argmap = {g["field"]: g["path"] for g in grouped}
            bids_argmap.update({s["field"]: s["path"] for s in static_results})

            arg_specs.append(
                ArgInputSpec(name=self.name,
                             argmap=bids_argmap,
                             bids_entities=bids_entities,
                             out_path=self.out_path,
                             method=self.method))

        return arg_specs


@dataclass
class ArgInputSpec:
    '''
    Class to record information about a
    set of files should be defined as inputs
    '''
    # TODO: add method to generate nipype node from args
    # Need a NodeFactory
    name: str
    argmap: dict = None
    bids_entities: tuple[tuple[str, str]]
    out_path: str
    method: str


def fetch_data(config, base_path):

    '''
    Helper function to provide a list of arguments
    given a configuration spec and base path
    '''

    cfg = SpecConfig(config)
    return cfg.get_file_args(base_path)
