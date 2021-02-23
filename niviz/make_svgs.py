from __future__ import annotations

import argparse
import niviz.node_factory
import niviz.config


def cli():
    '''
    CLI Entry function
    '''

    p = argparse.ArgumentParser(
        description="Command line interface to Niviz to generate "
        "QC images from pipeline outputs")

    p.add_argument('base_path', type=str, help="Base path to pipeline outputs")
    p.add_argument('spec_file',
                   type=str,
                   help="Specification configuration file")
    p.add_argument('out_path',
                   type=str,
                   help="Base output path to create SVGs")
    args = p.parse_args()

    arg_specs = niviz.config.fetch_data(args.spec_file, args.base_path)

    [
        niviz.node_factory.get_interface(a, args.out_path).run()
        for a in arg_specs
    ]
