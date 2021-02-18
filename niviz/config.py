"""
Contains classes/methods that validate and use a pipeline's
output configuration specification to generate arguments
for nipype ReportCapableInterfaces
"""

from __future__ import annotations
from typing import Optional

import os
import copy

import logging
import logging.config
import collections.abc
from string import Template
from pathlib import Path

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import re
from glob import glob
from itertools import groupby

from operator import itemgetter

from .node_factory import ArgInputSpec

logging.config.fileConfig("logging.conf")

# Initialize module logger
logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when Configuration File is incorrectly specified"""
    pass


def _nested_update(d: dict, u: dict) -> dict:
    '''
    Recursive updating of nested dict
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    '''
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# TODO use Path module
def _prefix_path(x: str, prefix: str) -> str:
    '''
    Prefix path with root directory
    '''

    if x.startswith("./"):
        return os.path.join(prefix, x.strip('.').strip('/'))
    else:
        return x


class SpecConfig(object):
    '''
    Class to provide interface to configuration
    specs for sourcing QC input files
    '''

    _yaml: Path
    defaults: dict
    file_specs: dict

    def __init__(self, yml: str, schema: str) -> None:

        # Validate yaml object and store original file
        with open(yml, 'r') as ystream:
            config = yaml.load(ystream, Loader=Loader)

        self._validate(yml, schema)
        self._yaml = Path(yml)

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
                for k, v in defaults['env'].items()
            }

        self.defaults = defaults

    def _substitute_env(self, env: str) -> str:
        '''
        Resolve system environment variables specified in global.env

        Note:
            All environment variables must be resolved

        Args:
            env: Strings in global.env containing environment variables

        Output:
            r: String with resolved environment variables

        Raises:
            ValidationError: If environment variables cannot be resolved
        '''

        r = os.path.expandvars(env)
        unresolved = re.findall("\\$[A-Za-z0-9]+", r)

        if unresolved:
            [
                logger.error(f"Undefined environment variable {u}!")
                for u in unresolved
            ]
            raise ValidationError
        return r

    def _validate(self, yml: dict, schema: dict) -> None:
        '''
        Validate YAML file

        Args:
            yml: Configuration specification
            schema: Schema definition to validate against

        Raises:
            ValidationError: If yml does not follow defined schema
        '''

        return

    def get_file_args(self, base_path: str) -> list[list[ArgInputSpec]]:
        '''
        Scrape `base_path` using configuration spec and construct
        arguments for image generation

        Args:
            base_path: Base path of outputs to scrape

        Returns:
            List of lists where each outer list defines a FileSpec entry
            and each inner-list defines a list of `ArgInputSpecs` used to
            generate an individual SVG image
        '''

        return [self._get_file_arg(f, base_path) for f in self.file_specs]

    def _get_file_arg(self, spec: dict, base_path: str) -> list[ArgInputSpec]:
        '''
        Construct argument for a single FileSpec

        Args:
            spec: Specification describing how to scrape files within
                pipeline outputs
            base_path: Root directory of pipeline outputs

        Returns:
            List of `ArgInputSpec` objects used to construct
            nipype.interfaces.mixins.ReportCapableInterface objects
        '''

        _spec = copy.deepcopy(spec)
        _spec['bids_map'] = _nested_update(spec['bids_map'],
                                           self.defaults.get('bids_map', {}))
        _spec['args'] = self._apply_envs(spec['args'])
        return FileSpec(_spec).gen_args(base_path)

    def _apply_envs(self, args: list[dict]) -> list[dict]:
        '''
        Apply specification global.env to values in dict

        Args:
            args: ReportCapableInterface to glob 'value' field with
                global variables to be substituted

        Returns:
            arg_list: ReportCapableInterface to 'value' field with
                global variables resolved
        '''

        if 'env' not in self.defaults:
            return args

        arg_list = []
        for f in args:

            try:
                f['value'] = Template(f['value']).substitute(
                    self.defaults['env'])
            except TypeError:
                if not isinstance(f['value'], bool):
                    logger.error("Unexpected value for argument "
                                 f"{f['field']} given value {f['value']}!")
                    raise

            arg_list.append(f)

        return arg_list


class FileSpec(object):
    '''
    Class to implement QcSpec
    '''
    def __init__(self, spec) -> None:

        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec['name']

    @property
    def method(self) -> str:
        return self.spec['method']

    @property
    def args(self) -> dict:
        return self.spec['args']

    # TODO: Implement args type
    def iter_args(self) -> tuple[str, str, bool]:
        '''
        Returns:

        A triple of
        (argument key, argument value, whether value is a BIDS field or not).
        Pulled from filespec[i].args in configuration spec
        '''
        for f in self.args:

            bids = f['no_bids'] if 'no_bids' in f else False
            yield (f['field'], f['value'], bids)

    @property
    def bids_map(self) -> dict:
        return self.spec['bids_map']

    @property
    def out_path(self) -> str:
        return self.spec['out_path']

    def _extract_bids_entities(self, path: str) -> tuple[tuple[str, ...], ...]:
        '''
        Extract BIDS entities from path

        Args:
            path: Input path for a filespec.args key

        Raises:
            ValueError: if all keys in bids_map cannot be found for a given
                path

        Returns:
            a tuple of BIDS (field,value) pairs
        '''

        # TODO: Handle standards where no BIDS values are to be found
        res = []
        raise_error = False
        for k, v in self.bids_map.items():

            if not isinstance(v, dict):
                logger.error("Config dict configured incorrectly!")
                logger.error(f"Key given is {v}")
                raise ValidationError

            bids_val = None
            if v.get('regex',False):
                try:
                    bids_val = re.search(v['value'], path)[0]
                except TypeError:
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

    def gen_args(self, base_path: Optional[str] = None) -> list[ArgInputSpec]:
        '''
        Constructs arguments used to build Nipype ReportCapableInterfaces
        using bids entities extracted from argument file paths and
        additional settings in configuration specification

        Args:
            base_path: Path to root directory of pipeline outputs

        Returns:
            List of arguments for a given filespec[i].args
        '''

        bids_results = []
        static_results = []
        for f, v, nobids in self.iter_args():
            if isinstance(v, str):
                v = _prefix_path(v, os.path.abspath(base_path))
            for p in glob(f"{v}"):

                cur_mapping = ({
                    "field": f,
                    "path": p,
                })

                if nobids:
                    static_results.append(cur_mapping)
                else:
                    bids_entities = self._extract_bids_entities(p)
                    bids_results.append((bids_entities, cur_mapping))

        matched = groupby(sorted(bids_results, key=itemgetter(0)), itemgetter(0))

        arg_specs = []
        for bids_entities, grouped in matched:

            bids_argmap = {g["field"]: g["path"] for _, g in grouped}
            bids_argmap.update({s["field"]: s["path"] for s in static_results})
            arg_spec = ArgInputSpec(name=self.name,
                             interface_args=bids_argmap,
                             bids_entities=bids_entities,
                             out_path=self.out_path,
                             method=self.method)

            arg_specs.append(arg_spec)

        return arg_specs


def fetch_data(config: str, base_path: str) -> list[ArgInputSpec]:
    '''
    Helper function to provide a list of arguments
    given a configuration spec and base path

    Args:
        config: Path to configuration specification
        base_path: Path to root directory of pipeline outputs

    Returns:
        List of `ArgInputSpec` to be used to automate construction
        of Nipype ReportCapableInterface objects
    '''

    # TODO: Implement schema
    cfg = SpecConfig(config, '')

    # Unnest output
    return [
            a for c in
            cfg.get_file_args(base_path)
            for a in c
    ]
