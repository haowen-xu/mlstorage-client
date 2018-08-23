import requests
from cachetools import LRUCache

from mlstorage_client.schema import (validate_experiment_id,
                                     validate_experiment_doc,
                                     validate_relpath)

__all__ = ['ApiClientV1']


class ApiClientV1(object):
    """
    Thin client binding for API v1.
    """

    def __init__(self, base_uri):
        """
        Construct a new :class:`ClientV1`.

        Args:
            base_uri (str): Base URI of the MLStorage server, e.g.,
                "http://example.com".
        """
        base_uri = base_uri.rstrip('/')
        self._base_uri = base_uri
        self._storage_dir_cache = LRUCache(128)

    def _update_storage_dir_cache(self, doc):
        self._storage_dir_cache[doc['id']] = doc['storage_dir']

    @property
    def base_uri(self):
        """Get the base URI of the MLStorage server."""
        return self._base_uri

    def do_request(self, method, endpoint, **kwargs):
        """
        Do `method` request against given `endpoint`.

        Args:
            method (str): The HTTP request method.
            endpoint (str): The endpoint of the API, should start with a
                slash "/".  For example, "/_query".
            \**kwargs: Arguments to be passed to :func:`requests.request`.

        Returns:
            The response object.
        """
        uri = self.base_uri + '/v1' + endpoint
        resp = requests.request(method, uri, **kwargs)
        if resp.status_code != 200:
            raise RuntimeError('HTTP error {}: {}'.
                               format(resp.status_code, resp.text))
        return resp

    def query(self, filter=None, skip=0, limit=10):
        ret = self.do_request(
            'POST', '/_query?skip={}&limit={}'.format(skip, limit),
            json=filter or {}).json()
        for doc in ret:
            self._update_storage_dir_cache(doc)
        return ret

    def get(self, id):
        id = validate_experiment_id(id)
        ret = self.do_request('GET', '/_get/{}'.format(id)).json()
        self._update_storage_dir_cache(ret)
        return ret

    def heartbeat(self, id):
        id = validate_experiment_id(id)
        return self.do_request(
            'POST', '/_heartbeat/{}'.format(id), data=b'').json()

    def create(self, name, doc_fields=None):
        doc_fields = dict(doc_fields or ())
        doc_fields['name'] = name
        doc_fields = validate_experiment_doc(doc_fields)
        ret = self.do_request('POST', '/_create', json=doc_fields).json()
        self._update_storage_dir_cache(ret)
        return ret

    def update(self, id, doc_fields):
        id = validate_experiment_id(id)
        doc_fields = validate_experiment_doc(dict(doc_fields))
        ret = self.do_request(
            'POST', '/_update/{}'.format(id), json=doc_fields).json()
        self._update_storage_dir_cache(ret)
        return ret

    def set_finished(self, id, status, doc_fields):
        id = validate_experiment_id(id)
        doc_fields = dict(doc_fields or ())
        doc_fields['status'] = status
        doc_fields = validate_experiment_doc(doc_fields)
        ret = self.do_request(
            'POST', '/_set_finished/{}'.format(id), json=doc_fields).json()
        self._update_storage_dir_cache(ret)
        return ret

    def delete(self, id):
        id = validate_experiment_id(id)
        ret = self.do_request(
            'POST', '/_delete/{}'.format(id), data=b'').json()
        for i in ret:
            self._storage_dir_cache.pop(i, None)
        return ret

    def get_storage_dir(self, id):
        id = str(validate_experiment_id(id))
        storage_dir = self._storage_dir_cache.get(id, None)
        if storage_dir is None:
            doc = self.get(id)
            storage_dir = doc['storage_dir']
        return storage_dir

    def getfile(self, id, path):
        id = str(validate_experiment_id(id))
        path = validate_relpath(path)
        return self.do_request(
            'GET', '/_getfile/{}/{}'.format(id, path)).content
