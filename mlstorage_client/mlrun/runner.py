import codecs
import json
import logging
import os
import shutil
import socket
import stat
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from threading import Thread, Condition
from logging import getLogger

import six

import mlstorage_client
from mlstorage_client.api_client_v1 import ApiClientV1
from mlstorage_client.utils import (JsonEncoder, exec_proc, run_tensorboard,
                                    clone_file_or_dir, compute_fs_size)

__all__ = ['run_experiment']


def get_environ_dict(client_args, id, storage_dir):
    """
    Get the actual environmental variables dict for executing the program.

    Args:
        client_args: The client arguments.
        id (str or ObjectId): The experiment ID.
        storage_dir (str): The storage directory.

    Returns:
        dict[str, str]: The environmental variables dict.
    """
    # inherit from the current environment, but remove the censored ones.
    env_dict = dict(os.environ)
    for forbid in ('MLSTORAGE_EXPERIMENT_ID', 'PWD'):
        if forbid in env_dict:
            env_dict.pop(forbid)

    # use the user provided environmental variables.
    env_dict.update(client_args.env)

    # set some variables corresponding to the experiment.
    env_dict['MLSTORAGE_SERVER_URI'] = str(client_args.server)
    env_dict['MLSTORAGE_EXPERIMENT_ID'] = str(id)
    env_dict['PWD'] = storage_dir

    # special treatment for Python programs: ensuring Python program's
    # output is not buffered.
    env_dict['PYTHONUNBUFFERED'] = '1'

    return env_dict


def get_creation_doc(client_args):
    """
    Get the document for creating an experiment according to `client_args`.

    Args:
        client_args: The client arguments.

    Returns:
        dict: The experiment document for creation.
    """
    doc = {
        'name': client_args.name,
        'args': list(client_args.args),
        'exc_info': {
            'hostname': socket.gethostname()
        }
    }
    for field in ('parent_id', 'description', 'tags', 'fingerprint', 'config'):
        if getattr(client_args, field):
            doc[field] = getattr(client_args, field)
    return doc


def log_dict(title, var_dict):
    """
    Dump `var_dict` in logs.

    Args:
        title (str): Title of this log.
        var_dict (dict): The dict to log.
    """
    if var_dict:
        getLogger(__name__).info(
            '%s:\n  %s', title, '\n  '.join([
                '{}={}'.format(k, v) for k, v in six.iteritems(var_dict)
            ])
        )


class CleanupHelper(object):
    """Class to help cleanup file entries."""

    def __init__(self):
        self._paths = set()

    def add(self, path):
        self._paths.add(path)

    def cleanup(self):
        paths = sorted(self._paths)
        self._paths = set()
        for path in paths:
            try:
                st = os.stat(path, follow_symlinks=False)
                if stat.S_ISDIR(st.st_mode):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except FileNotFoundError:
                pass
            except Exception:
                getLogger(__name__).info(
                    'Failed to cleanup: %s', path, exc_info=True)
                self._paths.add(path)


class CronJob(object):

    def __init__(self, name, interval, api, doc):
        self.name = name
        self.interval = interval
        self.api = api
        self.doc = doc
        self.stopped = False
        self.wait_cond = Condition()

    @property
    def id(self):
        return self.doc['id']

    @property
    def storage_dir(self):
        return self.doc['storage_dir']

    def _run_loop(self):
        while not self.stopped:
            try:
                self.run_once()
                getLogger(__name__).debug('Job thread %r executed.', self.name)
            except Exception as ex:
                getLogger(__name__).warning(
                    'Failed to execute job %r: %s', self.name, str(ex),
                    exc_info=True
                )
            with self.wait_cond:
                if not self.stopped:
                    if self.wait_cond.wait(self.interval):
                        break

    def run_once(self):
        raise NotImplementedError()

    @contextmanager
    def run_in_background(self):
        self.stopped = False
        thread = Thread(target=self._run_loop, daemon=True)
        try:
            thread.start()
            yield self
        finally:
            with self.wait_cond:
                self.stopped = True
                self.wait_cond.notify_all()
            thread.join()
            getLogger(__name__).debug('Job thread %r exited.', self.name)


class HeartbeatJob(CronJob):

    def __init__(self, name, api, doc):
        super(HeartbeatJob, self).__init__(name, 120, api, doc)

    def run_once(self):
        self.api.heartbeat(self.id)


class CollectJsonDictJob(CronJob):

    def __init__(self, name, api, doc, filename, field, postprocess=None):
        super(CollectJsonDictJob, self).__init__(name, 10, api, doc)
        self.filename = filename
        self.field = field
        self.postprocess = postprocess
        self.last_mtime = None
        self.last_size = None

    def run_once(self, force=False):
        path = os.path.join(self.storage_dir, self.filename)
        try:
            st = os.stat(path, follow_symlinks=True)
            if stat.S_ISREG(st.st_mode) and (force or
                                             st.st_mtime != self.last_mtime or
                                             st.st_size != self.last_size):
                with codecs.open(path, 'rb', 'utf-8') as f:
                    value = json.load(f)
                    if not isinstance(value, dict):
                        raise ValueError('JSON file content is not a dict')
                    else:
                        if self.postprocess:
                            value = self.postprocess(value)
                        self.api.update(self.id, {self.field: value})
                        self.doc[self.field] = value
                self.last_mtime = st.st_mtime
                self.last_size = st.st_size
        except FileNotFoundError:
            pass


class ConsoleDuplicator(object):

    def __init__(self, storage_dir, log_file):
        log_path = os.path.join(storage_dir, log_file)
        self.fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        self.dup_fd = sys.stdout.fileno()

    def __enter__(self):
        # drain the sys.stdout and sys.stderr buffers
        for io_file in (sys.stdout, sys.stderr):
            io_file.flush()
            if hasattr(io_file, 'buffer'):
                io_file.buffer.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def on_output(self, buf):
        os.write(self.fd, buf)
        os.write(self.dup_fd, buf)


@contextmanager
def maybe_run_tensorboard(client_args, api, doc):
    if client_args.tensorboard is not None:
        with run_tensorboard(doc['storage_dir']) as uri:
            webui = {'TensorBoard': uri}
            api.update(doc['id'], {'webui': webui})
            doc['webui'] = webui
            getLogger(__name__).info('TensorBoard launched at: %s', uri)
            yield uri
        getLogger(__name__).info('TensorBoard exited.')
    else:
        yield None


def retry(func, tag, wait_intervals=(10, 30, 60, 120, 300)):
    for itv in wait_intervals:
        try:
            return func()
        except Exception:
            getLogger(__name__).info(
                'Failed to %s, wait for %s seconds to retry', tag, itv)
        time.sleep(itv)
    return func()


def run_experiment(client_args):
    """
    Run the experiment with specified argument.

    Args:
        client_args: The client arguments.
    """
    # initialize logging and print the arguments to log
    logging.basicConfig(
        level='DEBUG' if client_args.debug else 'INFO',
        format='%(asctime)s [%(levelname)s]: %(message)s'
    )
    logger = getLogger(__name__)
    logger.info('MLStorage Runner {}'.format(mlstorage_client.__version__))
    if client_args.debug:
        logger.debug('DEBUGGING FLAG IS SET!')
    if client_args.parent_id:
        logger.info('Launched within a parent experiment: %s',
                    client_args.parent_id)
    log_dict('Environmental variables', client_args.env)
    log_dict('Experiment config', client_args.config)

    # establish connection to the server, and create the experiment
    api = ApiClientV1(client_args.server)
    doc = api.create(client_args.name, get_creation_doc(client_args))
    cleanup_helper = CleanupHelper()
    proc = None
    final_status_set = False

    try:
        # extract information from the returned doc
        id = doc['id']
        storage_dir = doc['storage_dir']
        logger.info('Experiment ID: %s', id)
        logger.info('Work-dir: %s', storage_dir)

        # ensure the shared network file system works, by generating a
        # random file, and try to get from the server
        os.makedirs(storage_dir, exist_ok=True)
        needle_fn = str(uuid.uuid4()) + '.txt'
        needle_path = os.path.join(storage_dir, needle_fn)
        needle_content = str(uuid.uuid4()).encode('utf-8')
        with open(needle_path, 'wb') as f:
            f.write(needle_content)
        remote_content = api.getfile(id, needle_fn)
        if remote_content != needle_content:
            raise ValueError('The content of remote file does not agree '
                             'with the generated content.')
        os.remove(needle_path)

        # construct the env dict
        env = get_environ_dict(client_args, id, storage_dir)

        # further update the doc according to `id` and `storage_dir`
        exc_info = doc.get('exc_info', {})
        exc_info.update({
            'work_dir': storage_dir,
            'env': env
        })
        doc = api.update(id, {'exc_info': exc_info})

        # prepare for the working directory
        for script_file in client_args.script_files:
            clone_file_or_dir(os.path.join(client_args.cwd, script_file),
                              script_file, storage_dir, symlink=False)
        if not client_args.no_link:
            for data_file in client_args.data_files:
                if data_file not in client_args.script_files:
                    clone_file_or_dir(os.path.join(client_args.cwd, data_file),
                                      data_file, storage_dir, symlink=True)
                    cleanup_helper.add(os.path.join(storage_dir, data_file))

        if client_args.config:
            config_json = json.dumps(client_args.config, cls=JsonEncoder,
                                     sort_keys=True, indent=2)
            with codecs.open(os.path.join(storage_dir, 'config.json'),
                             'wb', 'utf-8') as f:
                f.write(config_json)

        # scoped class for injecting TensorBoard webui
        class TensorBoardWebUI(object):
            def __init__(self, key='TensorBoard'):
                self.uri = None
                self.key = key

            def postprocess(self, webui):
                if self.uri:
                    webui[self.key] = self.uri
                else:
                    webui.pop(self.key, None)
                return webui

            @contextmanager
            def set_uri(self, uri):
                self.uri = uri
                try:
                    yield
                finally:
                    self.uri = None

        # run the program
        heartbeat_job = HeartbeatJob('send heartbeat', api, doc)
        default_config_job = CollectJsonDictJob(
            'collect default config', api, doc, filename='config.defaults.json',
            field='default_config'
        )
        result_job = CollectJsonDictJob(
            'collect result', api, doc, filename='result.json', field='result')
        tb_webui = TensorBoardWebUI()
        webui_job = CollectJsonDictJob(
            'collect webui', api, doc, filename='webui.json', field='webui',
            postprocess=tb_webui.postprocess
        )

        try:
            with maybe_run_tensorboard(client_args, api, doc) as tb_uri, \
                    tb_webui.set_uri(tb_uri), \
                    heartbeat_job.run_in_background(), \
                    default_config_job.run_in_background(), \
                    result_job.run_in_background(), \
                    webui_job.run_in_background(), \
                    ConsoleDuplicator(storage_dir, 'console.log') as out_dup, \
                    exec_proc(client_args.args,
                              on_stdout=out_dup.on_output,
                              stderr_to_stdout=True,
                              cwd=storage_dir,
                              env=env) as p:
                proc = p

                # final update the doc, to store pid
                exc_info = doc.get('exc_info', {})
                exc_info.update({
                    'pid': proc.pid
                })
                doc = api.update(id, {'exc_info': exc_info})

                p.wait()
                logger.debug('Process exited normally.')

        finally:
            # collect the JSON dict for the last time
            retry(lambda: webui_job.run_once(True),
                  webui_job.name)
            retry(lambda: default_config_job.run_once(True),
                  default_config_job.name)
            retry(lambda: result_job.run_once(True),
                  result_job.name)
            logger.debug('JSON file collected.')

            # cleanup the working directory
            cleanup_helper.cleanup()
            logger.debug('Working directory cleanup finished.')

            # compute the storage size and update the result
            if proc is not None:
                result_dict = {
                    'exit_code': proc.poll(),
                    'storage_size': compute_fs_size(storage_dir)
                }
                retry(lambda: api.set_finished(id, 'COMPLETED', result_dict),
                      'store the experiment result')
                final_status_set = True
                logger.info('Experiment exited with code: %s', proc.poll())

    except Exception as ex:
        if not final_status_set:
            logger.exception('Failed to run the experiment.')
            error_dict = {
                'message': str(ex),
                'traceback': ''.join(
                    traceback.format_exception(*sys.exc_info()))
            }
            retry(lambda: api.set_finished(doc['id'], 'FAILED',
                                           {'error': error_dict}),
                  'store the experiment failure')

    finally:
        cleanup_helper.cleanup()  # ensure to cleanup
        if client_args.debug:
            retry(lambda: api.delete(doc['id']), 'cleanup debugging experiment')
            logger.debug('Experiment deleted.')
