**THIS UTILITY HAS BEEN INCLUDED IN THE NEW ML-ESSENTIALS PROJECT**

`ML Essentials <https://github.com/haowen-xu/ml-essentials>`

MLStorage Client
================

The client side for MLStorage.
MLStorage is an application for running machine learning experiments
and storing results as well as generated files, all accessible with a dashboard.

The server side: `MLStorage Server <http://github.com/haowen-xu/mlstorage-server>`_.

Installation
------------

.. code-block:: bash

    pip install git+https://github.com/haowen-xu/mlstorage-client.git

Client Usage
------------

CLI Client
~~~~~~~~~~

.. code-block:: bash

    mlrun -s http://<ip>:8080 -- python train.py data.csv

The above command runs ``train.py`` in the experiment storage directory
(i.e., working directory) assigned by the server ``http://<ip>:8080``.
The script file, ``train,py``, will be copied to the storage directory.
Meanwhile, the data file, ``data.csv``, will be linked to that directory
(on Unix, a symbolic link is created; on Windows, a copied file is created).
The name of the experiment will automatically chosen to be ``train.py``.
