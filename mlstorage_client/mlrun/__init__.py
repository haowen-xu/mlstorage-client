import os
import sys
from collections import namedtuple
from urllib.parse import urlparse

import click
from mlstorage_client.schema import validate_experiment_id
from mlstorage_client.utils import (collect_relative_files,
                                    is_script_file,
                                    fingerprint_for_files,
                                    parse_tags,
                                    parse_config,
                                    parse_config_file,
                                    parse_env_file,
                                    parse_env)

from .runner import run_experiment

__all__ = ['mlrun']


TensorBoardArgs = namedtuple('TensorBoardArgs', ['host', 'port'])
ClientArgs = namedtuple(
    'EntryArgs',
    ['parent_id', 'name', 'description', 'tags', 'config', 'env', 'fingerprint',
     'server', 'no_link', 'debug', 'tensorboard', 'args',
     'cwd', 'script_files', 'data_files']
)


@click.command()
@click.option('-n', '--name', required=False, default=None,
              help='Experiment name.')
@click.option('-d', '--description', required=False, default=None,
              help='Experiment description.')
@click.option('-t', '--tags', required=False, multiple=True,
              help='Experiment tags, comma separated strings, e.g. '
                   '"prec 0.996, state of the arts".')
@click.option('-c', '--config', required=False, multiple=True,
              help='Configuration values, comma separated key-value pairs, '
                   'e.g., "max_epoch=1, normalizer=batch_normalization". '
                   'This will override "--config-file".')
@click.option('--config-file', help='Load configuration values from JSON file.',
              required=False, default=None)
@click.option('-e', '--env', required=False, multiple=True,
              help='Environmental variable (FOO=BAR). This will override '
                   '"--env-file".')
@click.option('--env-file', required=False,
              help='Load environmental variables from file.')
@click.option('--gpu', required=False, multiple=True,
              help='Quick approach to set the "CUDA_VISIBLE_DEVICES" '
                   'environmental variable.')
@click.option('-f', '--fingerprint', required=False, default=None,
              help='Specify the fingerprint for the experiment.')
@click.option('-s', '--server', required=True,
              default=os.environ.get('MLSTORAGE_SERVER_URI', '') or None,
              help='Specify the URI of MLStorage API server, e.g., '
                   '"http://localhost:8080".  If not specified, will use '
                   '``os.environ["MLSTORAGE_SERVER_URI"]``.')
@click.option('--no-link', is_flag=True, default=False, required=False,
              help='Do not link data files.')
@click.option('--tensorboard', is_flag=True, default=False, required=False,
              help='Run TensorBoard in the program\'s working directory.')
@click.option('--tensorboard-host', required=False, default=None,
              help='Specify the host of TensorBoard to bind.')
@click.option('--tensorboard-port', required=False, type=click.INT, default=0,
              help='Specify the port of TensorBoard.')
@click.option('--debug', is_flag=True, default=False, required=False,
              help='Whether or not to debug the program rather than run it '
                   'formally? If specified, the experiment will be deleted '
                   'after finished.')
@click.argument('args', nargs=-1)
def mlrun(name, description, tags, config, config_file, env, env_file,
          gpu, fingerprint, server, no_link,
          tensorboard, tensorboard_host, tensorboard_port, debug,
          args):
    """
    Run an experiment.

    The program arguments should be specified at the end, after a "--" mark,
    for example::

        mlrun --name "Experiment 1" -- python train.py

    The program will not run in the current directory.  Instead, it will run
    in the experiment storage directory, assigned by the server.
    If a program argument refers to an existing `script` file under the current
    directory (e.g., "train.py" in the above example), it will be copied
    to the experiment storage directory.  Otherwise if it refers other kinds
    of existing files or directories under the current directory, it will be
    `linked` to the storage directory (on Unix it uses symbolic links, while
    on Windows it just copies), unless `--no-link` is specified.
    Files outside the current directory will not be copied nor linked.
    You must use absolute paths to use such files.

    The following file extensions will be regarded as script files::

        *.py *.pl *.rb *.js *.sh *.r *.bat *.cmd *.exe *.jar

    If no `--name` argument is specified, the path of the first `script` file
    (relative to the current directory) will be used as the experiment name.
    If no `--fingerprint` argument is specified, a hashcode will be computed
    from all the `script` files as the program fingerprint.

    During the execution of the program, the STDOUT and STDERR of the program
    will be captured and stored in "console.log", under the experiment storage
    directory.

    Any configuration values specified via `--config` or `--config-file` will
    be serialized as a JSON file, stored as "config.json" under the experiment
    storage directory (i.e., the program working directory).

    The program may generate a "config.defaults.json" to inform the runner
    about the default configuration values, and it may also generate a
    "result.json" to inform the runner about the experiment results.
    Furthermore, it may also generate a "webui.json", to describe the URIs
    of the interactive web servers, which the program exposes to the user.
    The values from these files will be collected and stored into MongoDB.

    An example of "config.defaults.json"::

        {"max_epoch": 1000, "learning_rate": 0.01}

    An example of "result.json"::

        {"accuracy": 0.996}

    And an example of "webui.json"::

        {"TensorBoard": "http://[ip]:6006"}

    The layout of the experiment storage directory is shown as follows::

    \b
        .
        |-- config.json
        |-- config.defaults.json
        |-- console.log
        |-- result.json
        |-- webui.json
        |-- ... (copied and linked files)
        `-- ... (other files generated by the program)

    After the execution of the program, the linked files will be deleted
    from the experiment storage directory, while the copied `script` files will
    be left un-deleted.
    """
    # check the server
    parsed_server = urlparse(server)
    if parsed_server.scheme not in ('http', 'https'):
        click.echo('`server` must be HTTP or HTTPS uri: got {}'.format(server),
                   err=True)
        sys.exit(-1)

    # parse the parent id
    parent_id = os.environ.get('MLSTORAGE_EXPERIMENT_ID') or None
    if parent_id is not None:
        try:
            parent_id = str(validate_experiment_id(parent_id))
        except ValueError:
            click.echo('os.environ["MLSTORAGE_EXPERIMENT_ID"] is invalid: {!r}'.
                       format(parent_id))

    # parse the arguments, find out script files and other files
    if not args:
        click.echo('You must specify program arguments.', err=True)
        sys.exit(-1)
    cwd = os.getcwd()
    arg_files = collect_relative_files(args, cwd)
    arg_scripts, arg_datafiles = [], []
    for arg_file in arg_files:
        if is_script_file(arg_file, cwd):
            arg_scripts.append(arg_file)
        else:
            arg_datafiles.append(arg_file)

    # choose a name if not specified
    if name is None:
        if not arg_scripts:
            click.echo('Cannot infer the experiment name: no script file '
                       'found.', err=True)
            sys.exit(-1)
        name = arg_scripts[0]  # including the relative path

    # parse the tags
    merged_tags = []
    for tag_text in tags:
        for tag in parse_tags(tag_text):
            if tag not in merged_tags:
                merged_tags.append(tag)

    # parse the config
    config_dict = {}
    if config_file:
        config_dict.update(parse_config_file(config_file))
    if config:
        for config_text in config:
            config_dict.update(parse_config(config_text))

    # parse the env
    env_dict = {}
    if env_file is not None:
        env_dict.update(parse_env_file(env_file))
    for env_text in env:
        k, v = parse_env(env_text)
        env_dict[k] = v
    if gpu:
        gpu_list = []
        for gpu_text in gpu:
            gpu_list.extend(filter(lambda s: s, gpu_text.split(',')))

        def gpu_sort_key_func(g):
            try:
                return '', int(g)
            except (TypeError, ValueError):
                return g, 0

        gpu_list = sorted(set(gpu_list), key=gpu_sort_key_func)
        env_dict['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))

    # parse the fingerprint
    fingerprint = fingerprint or None
    if not fingerprint:
        fingerprint = fingerprint_for_files(arg_scripts, cwd)

    # parse the tensorboard args
    if tensorboard:
        tb_args = TensorBoardArgs(tensorboard_host, tensorboard_port)
    else:
        tb_args = None

    # assemble the argument object
    client_args = ClientArgs(
        parent_id=parent_id, name=name, description=description,
        tags=merged_tags, config=config_dict, env=env_dict,
        fingerprint=fingerprint, server=server, no_link=no_link,
        debug=debug, tensorboard=tb_args, args=args,
        cwd=cwd, script_files=set(arg_scripts), data_files=set(arg_datafiles),
    )
    run_experiment(client_args)


if __name__ == '__main__':
    mlrun()
