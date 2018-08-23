MLStorage
=========

A pair of server and client applications, to run machine learning experiments
and store results as well as generated files, all accessible with a dashboard.

Pre-requisites
--------------

*   Python: >= 3.5.3, or Docker (for server)
*   MongoDB: for storing experiment documents.
*   A shared network file system: currently the programs must run on a host
    where the server's storage directory is accessible in the same location,
    so as to store its generated files.

Server Usage
------------

Install via Pip
~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install git+https://github.com/haowen-xu/mlstorage.git

    mlboard -h <ip> -p 8080 -w 4 -R /path/to/storage-dir \
        -M mongodb://user:password@localhost/admin -D user -C experiments


The above command starts an MLStorage server at ``http://<ip>:8080``, with
``4`` workers to serve requests.  The MongoDB connection string is set to
``mongodb://user:password@localhost/admin``, with ``user`` and ``password`` as
the login credential.  The ``user`` database and the ``experiments`` collection
is chosen to store the experiment documents.  The root directory of experiment
storage directory (i.e., working directory) is set to ``/path/to/storage-dir``.

Install from Docker
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker build \
        --build-arg UBUNTU_MIRROR=archive.ubuntu.com \
        --build-arg TZ=UTC \
        -t haowen-xu/mlserver \
        mlboard-docker

    docker run \
        --name mlserver -d \
        -p 8080 \
        -e MLSTORAGE_SERVER_HOST=0.0.0.0 \
        -e MLSTORAGE_SERVER_PORT=8080 \
        -e MLSTORAGE_SERVER_WORKERS=4 \
        -e MLSTORAGE_EXPERIMENT_ROOT=/path/to/experiments \
        -v /path/to/experiments:/path/to/experiments \
        -e MLSTORAGE_MONGO_CONN=mongodb://localhost:27017 \
        -e MLSTORAGE_MONGO_DB=test \
        -e MLSTORAGE_MONGO_COLL=experiments \
        haowen-xu/mlserver


Client Usage
------------

CLI Client
~~~~~~~~~~

.. code-block:: bash

    pip install git+https://github.com/haowen-xu/mlstorage.git

    mlrun -s http://<ip>:8080 -- python train.py data.csv

The above command runs ``train.py`` in the experiment storage directory
(i.e., working directory) assigned by the server ``http://<ip>:8080``.
The script file, ``train,py``, will be copied to the storage directory.
Meanwhile, the data file, ``data.csv``, will be linked to that directory
(on Unix, a symbolic link is created; on Windows, a copied file is created).
The name of the experiment will automatically chosen to be ``train.py``.


Other Clients
~~~~~~~~~~~~~

*   `MLComp <https://github.com/haowen-xu/mlcomp>`_: a Python class
    ``mlcomp.persist.Experiment`` to register the current running program as
    an experiment.  The program will run in its original working directory,
    requiring the user to determine which files should be copied to the
    experiment's working directory on the shared network file system.
