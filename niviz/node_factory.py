"""
Provides interface to join objects generated by ConfigSpec
to Nipype ReportCapableInterface

Provides the niviz.node_factory.factory object which can be used register and
create interfaces
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
if TYPE_CHECKING:
    from nipype.interfaces.mixins import reporting

import os
from pathlib import Path
from string import Template
from dataclasses import dataclass, InitVar
from copy import deepcopy

import logging
import logging.config

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("nodeFactory")


@dataclass
class ArgInputSpec:
    '''
    Store configuration options and method key for constructing
    Nipype ReportCapableInterface classes

    Args:
        out_spec: Template string for building output path
        bids_entities: BIDS entities (key,value) paired tuples

    Attributes:
        name: Name of SVG report being generated
        method: Method key for to select which
            `reporting.ReportCapableInterface` to select
        interface_args: Dictionary of **kwargs to pass to
            `reporting.ReportCapableInterface`
        bids_output: Path to target SVG output
    '''
    out_spec: InitVar[str]
    bids_entities: InitVar[tuple[tuple[str, str]]]

    name: str
    method: str
    interface_args: dict
    _out_spec: Path = None

    def __post_init__(self, out_spec, bids_entities):
        '''
        Construct final output path

        Raises:
            KeyError if BIDS entities required for output_path are missing
        '''

        self._out_spec = Path(
            Template(out_spec).substitute(
                {x[0]: f"{x[0]}-{x[1]}"
                 for x in bids_entities}))

    def make_interface_args(self,
                            out_path: Union[str, Path],
                            make_dirs: Optional[bool] = False) -> dict:
        '''
        Returns a dictionary to construct a ReportCapableInterface

        Args:
            out_path:   Path to root directory for outputting SVGs
            make_dirs:  Create final output directory

        Returns:
            kwargs to build ReportCapableInterface
        '''

        if not isinstance(out_path, Path):
            out_path = Path(out_path)

        interface_args = deepcopy(self.interface_args)
        out_dir = out_path / self._out_spec

        if make_dirs:
            os.makedirs(out_dir.parent, exist_ok=True)

        interface_args.update({'out_report': out_dir})
        return interface_args


class RPTFactory(object):
    '''
    Factory class to generate Nipype RPT nodes
    given argument specification objects derived
    from niviz

    Attributes:
        _interfaces: Mapping method strings to
            `reporting.ReportCapableInterface` subclass
    '''

    _interfaces: dict[str, reporting.ReportCapableInterface]

    def __init__(self):
        self._interfaces = {}

    def get_interface(
            self,
            spec: ArgInputSpec,
            out_path: Union[str, Path],
            make_dirs: Optional[bool] = False
    ) -> reporting.ReportCapableInterface:
        '''
        Retrieve and configure interface from registered list

        Args:
            spec: Input specification for generating SVG image
            out_path: Output path to store SVG files

        Returns:
            interface: Configured `reporting.ReportCapableInterface`
        '''

        try:
            interface_class = self._interfaces[spec.method]
        except KeyError:
            logger.error(
                f"View method {spec.method} has not been registered "
                "with RPTFactory. If using a custom ReportCapableInterface"
                " register to RPTFactory using\n"
                "from niviz.node_factory import register_interface\n"
                f"register_interface(ReportCapableInterface, {spec.method})")
            raise

        # Create and configure node args
        interface_args = spec.make_interface_args(out_path, make_dirs)
        logger.debug(
            f"Constructing {spec.name} using {interface_class}")
        logger.debug(f"Args:\n {interface_args}")
        return interface_class(generate_report=True, **interface_args)

    def register_interface(self,
                           rpt_interface: reporting.ReportCapableInterface,
                           method: str,
                           override: Optional[bool] = False) -> None:
        '''
        Register a RPT to enable creation with factory_method()

        Args:
            rpt_interface: `reporting.ReportCapableInterface` subclass
            method: String key for accessing interface
            override: Override an existing key if it exists

        Raises:
            KeyError: If method key already exists for another Interface
        '''

        if (method not in self._interfaces) or override:
            self._interfaces[method] = rpt_interface
        else:
            logger.error(
                f"Method already registered as {self._interfaces[method]}. "
                " Use override=True to replace existing method key")
            raise KeyError
        return

    def view_interfaces(self) -> dict[str, reporting.ReportCapableInterface]:
        '''
        Return a mapping of currently registered interfaces

        Returns:
            registered_interfaces: Dictionary of registered interfaces
        '''
        return self._interfaces


factory = RPTFactory()


def register_interface(rpt_interface: reporting.ReportCapableInterface,
                       method: str) -> None:
    '''
    Helper function to register a ReportCapableInterface to the persistent
    RPTFactory instance.

    Calls RPTFactory().register_interface(*args)

    Args:
        rpt_interface: Nipype ReportCapableInterface class
        method: String key for accessing Interface

    Raises:
        KeyError: If method key already exists for another Interface
    '''
    factory.register_interface(rpt_interface, method)


def get_interface(
        spec: ArgInputSpec,
        out_path: Union[str, Path],
        make_dirs: Optional[bool] = True) -> reporting.ReportCapableInterface:
    '''
    Generate Nipype ReportCapableInterfaces from a list of
    ArgInputSpecs

    Args:
        spec:  ArgInputSpec containing instructions for
        building visualization interfaces

    Returns:
        Iterable of Nipype ReportCapableInterfaces
    '''
    return factory.get_interface(spec, out_path, make_dirs=make_dirs)


# Avoid circular import problem
def initialize_defaults():
    import niviz.interfaces.views
    niviz.interfaces.views._run_imports()


initialize_defaults()
