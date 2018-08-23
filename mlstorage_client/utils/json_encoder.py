from datetime import datetime
import json

from bson import ObjectId
from pytz import UTC

__all__ = ['JsonEncoder']


class JsonEncoder(json.JSONEncoder):
    """
    Extended JSON encoder for serializing experiment documents.
    """

    def __init__(self, use_timestamp=False, **kwargs):
        super(JsonEncoder, self).__init__(**kwargs)
        self.use_timestamp = use_timestamp

    def _default_object_handler(self, o):
        if isinstance(o, datetime):
            if self.use_timestamp:
                # we only use UTC datetime through out this project
                yield o.replace(tzinfo=UTC).timestamp()
            else:
                yield o.isoformat()
        elif isinstance(o, ObjectId):
            yield str(o)
        elif isinstance(o, bytes):
            try:
                yield o.decode('utf-8')
            except UnicodeDecodeError:
                yield repr(o)

    #: List of object serialization handlers
    OBJECT_HANDLERS = [_default_object_handler]

    def default(self, o):
        for handler in self.OBJECT_HANDLERS:
            for obj in handler(self, o):
                return obj
        return super(JsonEncoder, self).default(o)

    def encode(self, o):
        return super(JsonEncoder, self).encode(o)
