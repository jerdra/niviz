from __future__ import annotations

import os
import argparse
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def _get_package_name(config):
    '''
    Get package name from YAML specification or report file
    '''

    with open(config, 'r') as f:
        package_name = yaml.load(f, Loader=Loader)['package']
    return package_name


def svg_util(args):
    '''
    SVG sub-command
    '''

    import niviz.node_factory
    import niviz.config

    arg_specs = niviz.config.fetch_data(args.spec_file, args.base_path)
    out_path = os.path.join(args.out_path, _get_package_name(args.spec_file))

    [niviz.node_factory.get_interface(a, out_path).run() for a in arg_specs]
    return


def report_util(args):
    '''
    Report generation utility
    '''

    from niworkflows.reports.core import run_reports

    package_path = os.path.join(args.base_path, _get_package_name(args.config))

    subject_list = args.subjects
    if args.subjects:
        subject_list = args.subjects
    else:
        subject_list = [
            s for s in os.listdir(package_path)
            if ('sub-' in s) and ('.html' not in s)
        ]

    [
        run_reports(args.output_dir,
                    s,
                    'NULL',
                    config=args.config,
                    reportlets_dir=args.base_path) for s in subject_list
    ]

    return


def cli():
    '''
    CLI Entry function
    '''

    p = argparse.ArgumentParser(
        description='Command line interface to Niviz to generate '
        'QC images from pipeline outputs')

    sub_parsers = p.add_subparsers(help='Niviz command modes')

    parser_svg = sub_parsers.add_parser('svg', help='SVG Generation utility')
    parser_svg.add_argument('base_path',
                            type=str,
                            help='Base path to pipeline outputs')
    parser_svg.add_argument('spec_file',
                            type=str,
                            help='Specification configuration file')
    parser_svg.add_argument('out_path',
                            type=str,
                            help='Base output path to create SVGs')
    parser_svg.set_defaults(func=svg_util)

    parser_report = sub_parsers.add_parser('report',
                                           help='Report Generation utility')
    parser_report.add_argument('base_path', type=str, help='Base path to SVGs')
    parser_report.add_argument('config',
                               type=str,
                               help='Path to report configuration')
    parser_report.add_argument('output_dir',
                               type=str,
                               help='Path to output reports')
    parser_report.add_argument('--subjects',
                               nargs='+',
                               help='List of subjects to generate reports for')
    p.set_defaults(func=report_util)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    cli()
