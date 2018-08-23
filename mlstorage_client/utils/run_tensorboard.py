import os
import re
import socket
from contextlib import contextmanager
from queue import Queue, Empty

from mlstorage_client.utils import exec_proc

__all__ = ['run_tensorboard']


@contextmanager
def run_tensorboard(path, log_file=None, host='0.0.0.0', port=0, timeout=30):
    """
    Run TensorBoard in background.

    Args:
        path (str): The log directory for TensorBoard.
        log_file (str): If specified, will write the log of TensorBoard
            into this file. (default :obj:`None`)
        host (str): Bind TensorBoard to this host. (default "0.0.0.0")
        port (int): Bind TensorBoard to this port. (default 0, any free port)
        timeout (float): Wait the TensorBoard to start for this number of
            seconds. (default 30)

    Yields:
        str: The URI of the launched TensorBoard.
    """
    def capture_output(data, fout, headbuf,
                       pattern=re.compile(br'TensorBoard \S+ at '
                                          br'http://([^:]+):(\d+)')):
        if headbuf:
            headbuf[0] = headbuf[0] + data
            m = pattern.search(headbuf[0])
            if m:
                url_host = m.group(1).decode('utf-8')
                url_port = m.group(2).decode('utf-8')
                if not url_host or (url_host in ('0.0.0.0', '::0')):
                    url_host = socket.gethostbyname(socket.gethostname())
                the_url = 'http://{}:{}'.format(url_host, url_port)
                url_q.put(the_url)
                del headbuf[:]
        if fout is not None:
            fout.write(data)
            fout.flush()

    @contextmanager
    def maybe_open_log():
        if log_file is not None:
            with open(log_file, 'wb') as f:
                yield f
        else:
            yield None

    url_q = Queue()
    args = ['tensorboard',
            '--logdir', path,
            '--host', host,
            '--port', str(port)]
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = ''  # force not using GPUs
    with maybe_open_log() as log_f, \
            exec_proc(args,
                      on_stdout=lambda data: capture_output(
                          data, fout=log_f, headbuf=[b'']
                      ),
                      stderr_to_stdout=True,
                      env=env):
        try:
            url = url_q.get(timeout=timeout)
        except Empty:
            raise ValueError('TensorBoard did not report its url in '
                             '{} seconds.'.format(timeout))
        yield url
