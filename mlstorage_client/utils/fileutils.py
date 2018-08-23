import hashlib
import os
import re
import shutil
import stat
import sys
from logging import getLogger

from mlstorage_client.schema import validate_relpath

__all__ = [
    'copy_or_link',  'clone_file_or_dir',
    'collect_relative_files', 'is_script_file',
    'fingerprint_for_files', 'compute_fs_size',
]

_SCRIPT_FILE_PATTERN = re.compile(
    r'.*\.(py|pl|rb|js|sh|r|bat|cmd|exe|jar)$',
    flags=re.IGNORECASE
)


def copy_or_link(source, target, symlink=True):
    """
    Copy a file, or link a file (using symbolic link), according to the
    argument `symlink` and the platform capabilities.

    Args:
        source (str): The source file.
        target (str): The target file or link.
        symlink (bool): Whether or not to use symbolic link if possible?
            (default :obj:`True`)

    Returns:

    """
    if symlink and sys.platform != 'win32':
        os.symlink(source, target)
    else:
        shutil.copy(source, target)


def clone_file(source_file, target_path, work_dir, symlink=True):
    """
    Clone a file to the program's working directory.

    Args:
        source_file (str): Path of the source file.
        target_path (str): Relative path of the target file under
            the program's working directory.
        work_dir (str): The working directory for the program.
        symlink: Whether to use symbolic link if possible.
            (default :obj:`True`)
    """
    full_target_path = os.path.join(work_dir, target_path)
    parent_dir = os.path.split(full_target_path)[0]
    os.makedirs(parent_dir, exist_ok=True)
    copy_or_link(source_file, full_target_path, symlink=symlink)


_CLONE_SKIP_PATTERN = re.compile(
    r'^(\.git|\.svn|\.cvs|\.hg|\.DS_Store|\.directory|Thumbs\.db)$')


def clone_dir(source_dir, target_path, work_dir, symlink=True):
    """
    Clone a directory to the program's working directory.

    Args:
        source_dir (str): Path of the source directory.
        target_path (str): Relative path of the target directory under
            the program's working directory.
        work_dir (str): The working directory for the program.
        symlink: Whether to use symbolic link if possible.
            (default :obj:`True`)
    """
    for name in os.listdir(source_dir):
        if _CLONE_SKIP_PATTERN.match(name):
            continue
        src_path = os.path.join(source_dir, name)
        dst_path = os.path.join(target_path, name)
        dst_fullpath = os.path.join(work_dir, dst_path)
        if os.path.isdir(src_path):
            os.makedirs(dst_fullpath, exist_ok=True)
            clone_dir(src_path, dst_path, work_dir, symlink)
        else:
            copy_or_link(src_path, dst_fullpath, symlink=symlink)
            getLogger(__name__).debug('Cloned file: %s', src_path)


def clone_file_or_dir(source_path, target_path, work_dir, symlink=True):
    """
    Clone a file or a directory to the program's working directory.

    Args:
        source_path (str): Path of the source file or directory.
        target_path (str): Relative path of the target file or directory
            under the program's working directory.
        work_dir (str): The working directory for the program.
        symlink: Whether to use symbolic link if possible.
            (default :obj:`True`)
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(source_path)
    elif os.path.isdir(source_path):
        os.makedirs(os.path.join(work_dir, target_path), exist_ok=True)
        clone_dir(source_path, target_path, work_dir, symlink)
    else:
        clone_file(source_path, target_path, work_dir, symlink)


def collect_relative_files(args, start_path):
    """
    Collect relative files or directories from `args` under `start_path`.

    Args:
        args (list[str]): List of arguments, potentially relative
            files or directories under `start_path`.
        start_path (str): The root directory.

    Returns:
        list[str]: The relative paths of the discovered files or directories.
    """
    start_path = os.path.abspath(start_path)
    ret = []
    for arg in args:
        arg_path = os.path.abspath(arg)
        try:
            arg_relpath = validate_relpath(
                os.path.relpath(arg_path, start_path))
        except ValueError:
            pass
        else:
            if os.path.exists(arg_path):  # for both files and directories
                ret.append(arg_relpath)
    return ret


def is_script_file(path, start_path):
    """
    Check whether or not `path` under `start_path` is a script file.

    Args:
        path (str): The path to be checked.
        start_path (str): The root directory.

    Returns:
        bool: Whether or not `path` is a script file.
    """
    return _SCRIPT_FILE_PATTERN.match(path) is not None and \
        os.path.isfile(os.path.join(start_path, path))


def fingerprint_for_files(file_list, root_dir, algorithm=hashlib.sha1,
                          buffer_size=16*1024):
    """
    Compute the fingerprint for specified `file_list`.

    Args:
        file_list (Iterable[str]): List of relative paths of the files.
            These paths will also be used in computing the fingerprint,
            so they should be normalized (i.e., use "/" instead of "\\"
            on Windows, eliminate "." and reduce ".." to minimal form).
        root_dir (str): The root directory of all the files.
        algorithm: The hash algorithm. (default ``hashlib.sha1``)
        buffer_size: Size of IO buffer. (default ``16*1024``)

    Returns:
        str: The computed fingerprint, or None if `file_list` is empty.
    """
    file_list = sorted(file_list)
    if not file_list:
        return None

    h = algorithm()
    for name in file_list:
        path = os.path.join(root_dir, name)
        st = os.stat(path, follow_symlinks=True)

        # consume file header
        head = '|'.join([str(v) for v in (len(path), path, st.st_size)])
        if not isinstance(head, bytes):
            head = head.encode('utf-8')
        h.update(head)

        # consume file content
        with open(path, 'rb') as f:
            while True:
                buf = f.read(buffer_size)
                if not buf:
                    break
                h.update(buf)
    return h.hexdigest()


def compute_fs_size(path):
    """
    Sum up the file system size of `path`.

    Args:
        path (str): The path to be analyzed.

    Returns:
        int: The size of `path` in bytes.
    """
    st = os.stat(path, follow_symlinks=False)
    if stat.S_ISDIR(st.st_mode):
        return st.st_size + sum(
            compute_fs_size(os.path.join(path, name))
            for name in os.listdir(path)
        )
    else:
        return st.st_size
