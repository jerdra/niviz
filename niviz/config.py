"""
Contains classes/methods that validate and use a pipeline's
output configuration specification to generate arguments
for nipype ReportCapableInterfaces
"""

from __future__ import annotations
from typing import Union, Optional

import os
import copy

import logging
import collections.abc
from string import Template
from pathlib import Path
from collections import defaultdict

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import re
from glob import glob

from .node_factory import ArgInputSpec

# Initialize module logger
logger = logging.getLogger("config")
if (logger.hasHandlers()):
    logger.handlers.clear()


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

    def _update_spec_with_defaults(self, spec):
        '''
        Update a single file specification with the global defaults if
        fields have not yet been defined

        Arguments:
            spec: file specification key value pairs

        Returns:
            Dictionary with defaults applied to spec
        '''
        _spec = copy.deepcopy(spec)
        _spec['bids_map'] = _nested_update(spec['bids_map'],
                                           self.defaults.get('bids_map', {}))

        _spec['bids_hierarchy'] = spec.get(
            'bids_hierarchy', self.defaults.get('bids_hierarchy', []))

        _spec['args'] = self._apply_envs(spec['args'])
        return _spec

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

        _spec = self._update_spec_with_defaults(spec)
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
    def __init__(self, spec: dict) -> None:

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
            ispath = f['path'] if 'path' in f else False
            yield (f['field'], f['value'], ispath)

    @property
    def bids_map(self) -> dict:
        return self.spec['bids_map']

    @property
    def static_entities(self) -> dict:
        static_entities = {
            k: v['value']
            for k, v in self.spec['bids_map'].items()
            if not v.get('regex', False)
        }
        return static_entities

    @property
    def dynamic_entities(self) -> dict:
        dyn_entities = {
            k: v['value']
            for k, v in self.spec['bids_map'].items() if v.get('regex')
        }
        return dyn_entities

    @property
    def out_path(self) -> str:
        return self.spec['out_path']

    def _extract_bids_entities(
            self,
            path: Optional[str] = None
    ) -> tuple[dict[str, Union[str, None]], ...]:
        '''
        Extract BIDS entities from path

        Args:
            path: Input path for a filespec.args key

        Raises:
            ValueError: if all keys in bids_map cannot be found for a given
                path

        Returns:
            a tuple of BIDS (field,value) pairs extracted from path
            a tuple of BIDS (field,value) pairs that were statically specified
        '''

        res = {}
        for k, v in self.dynamic_entities.items():
            try:
                bids_val = re.search(v, path)[0]
            except TypeError:
                logger.warning(f"Cannot extract {k} from {path} using {v}!")
                bids_val = None
            finally:
                res.update({k: bids_val})

        if all([v is None for v in {**res, **self.static_entities}.values()]):
            raise ValueError(f"No BIDS entities specified for {path}!")

        return res

    def _group_by_hierarchy(self, entities_specs, available_entities):
        '''
        Perform hierarchical grouping recursively
        '''
        def group_by_entity(entity_dict, entity):

            no_entity = []
            entity_found = defaultdict(list)
            for e, f in entity_dict:
                if e[entity] is None:
                    no_entity.append((e, f))
                else:
                    entity_found[e[entity]].append((e, f))
            return entity_found, no_entity

        def resolve_group(grouped_entities, match_entities):
            '''
            Resolve grouped entities from

            {"entity_value": [(entity_spec, file_spec),...], ...}

            into

            {tuple(match_entities): [file_spec,...], ...}

            Entities not in match_entities are discarded
            '''

            res = defaultdict(list)
            for g in grouped_entities.values():
                for e, s in g:
                    entities = {
                        k: v
                        for k, v in e.items() if k in match_entities
                    }
                    res[tuple(entities.items())].append(s)
            return res

        def apply_spread(res, spread):
            '''
            Apply items with no hierarchy
            to results
            '''
            if not spread:
                return res

            [res[k].append(f) for k in res.keys() for _, f in spread]

            return res

        # TODO: Update to use indices to avoid copying
        def traverse(h, entity_specs):

            # Bind to local function scope hierarchy for read-only
            nonlocal hierarchy

            entity = h.pop(0)
            has_entity, to_spread = group_by_entity(entity_specs, entity)

            if not h:
                return apply_spread(resolve_group(has_entity, hierarchy),
                                    to_spread)

            res = {}
            for es in has_entity.values():
                c_res = traverse(copy.copy(h), es)
                c_res = apply_spread(c_res, to_spread)
                res.update(c_res)

            return res

        hierarchy = [
            h for h in self.spec['bids_hierarchy'] if h in available_entities
        ]
        return traverse(copy.copy(hierarchy), entities_specs)

    def gen_args(self, base_path: str) -> list[ArgInputSpec]:
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
        for f, v, ispath in self.iter_args():

            if isinstance(v, str) and ispath:
                v = _prefix_path(v, os.path.abspath(base_path))
                for p in glob(f"{v}"):

                    cur_mapping = ({
                        "field": f,
                        "value": p,
                    })

                    entities = self._extract_bids_entities(p)
                    bids_results.append((entities, cur_mapping))
            else:
                entities = self._extract_bids_entities()
                bids_results.append((entities, {"field": f, "value": v}))

        matched = self._group_by_hierarchy(bids_results, self.bids_map.keys())
        arg_specs = []
        for bids_entities, filespecs in matched.items():

            bids_argmap = {i["field"]: i["value"] for i in filespecs}
            arg_spec = ArgInputSpec(name=self.name,
                                    interface_args=bids_argmap,
                                    bids_entities=(
                                        *bids_entities,
                                        *tuple(self.static_entities.items())
                                    ),
                                    out_spec=self.out_path,
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
    return [a for c in cfg.get_file_args(base_path) for a in c]
