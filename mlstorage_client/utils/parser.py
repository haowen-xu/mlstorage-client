import codecs
import json
import re

import pyparsing as pp

__all__ = [
    'Tokens',
    'parse_config', 'parse_config_file',
    'parse_tags',
    'parse_env', 'parse_env_file'
]


class Tokens(object):
    # shared tokens
    delim_chars = '[]{},'
    pause = pp.FollowedBy(pp.Word(delim_chars) | pp.StringEnd())
    number = (pp.pyparsing_common.number + pause)
    quoted_string = pp.QuotedString('"', escChar='\\')
    true = (pp.Regex(r'(True|true|yes|on)') + pause). \
        setParseAction(lambda _: True)
    false = (pp.Regex(r'(False|false|no|off)') + pause). \
        setParseAction(lambda _: False)
    unquoted_string = pp.CharsNotIn(delim_chars). \
        setParseAction(lambda toks: toks[0].strip())
    empty_value = pp.Empty(). \
        setParseAction(lambda _: '')

    # tokens for configs
    identifier = pp.pyparsing_common.identifier.copy()
    comma = pp.Literal(',').suppress()
    assign = pp.Literal('=').suppress()
    config_value = (
        number | true | false | quoted_string | unquoted_string | empty_value)
    key_value_pair = (identifier + assign + config_value). \
        setParseAction(lambda toks: (toks[0], toks[1]))
    key_value_pair_list = pp.Optional(
        key_value_pair + pp.ZeroOrMore(comma + key_value_pair))

    # tokens for tags
    tag = quoted_string | unquoted_string | pp.Empty().suppress()
    tag_list = pp.Optional(tag + pp.ZeroOrMore(comma + tag))


def parse_config(config_text):
    """
    Parse configuration text like ``name1=value1,name2=value2`` into
    dict ``{'name1': 'value1', 'name2': 'value2'}``.

    Args:
        config_text (str): The text to be parsed.

    Returns:
        dict[str, any]: The parsed configuration dict.
    """
    t = Tokens.key_value_pair_list
    return dict(t.parseString(config_text, parseAll=True).asList())


def parse_config_file(config_file):
    """
    Parse configuration values from a JSON file.

    Args:
        config_file (str): Path of the JSON file.

    Returns:
        dict[str, any]: The parsed configuration dict.
    """
    with codecs.open(config_file, 'rb', 'utf-8') as f:
        config_dict = json.loads(f.read().strip())
        if not isinstance(config_dict, dict):
            raise ValueError('Config file does not contain a dict: {}'.
                             format(config_file))
    return config_dict


def parse_tags(tags_text):
    """
    Parse tags text like ``tag1, tag2`` into list ``['tag1', 'tag2']``.

    Args:
        tags_text (str): The text to be parsed.

    Returns:
        list[str]: The parsed tags.
    """
    t = Tokens.tag_list
    return [s for s in t.parseString(tags_text, parseAll=True).asList() if s]


_KV_PATTERN = re.compile(r'^\s*([^=]+?)\s*=\s*(.*?)\s*$')
_KV_PATTERN_STRICT = re.compile(r'([^=]+)=(.*)$')


def parse_env(source, ignore_spaces=True):
    """
    Parse the environmental variable from "FOO=BAR".

    Args:
        source (str): The source text to be parsed.
        ignore_spaces (bool): Whether or not to ignore spaces at the front
            and tail of `source`, and surrounding "="?  (default :obj:`True`)

    Returns:
        (str, str): Tuple of ("FOO", "BAR").

    Raises:
        ValueError: If the source has syntax error.
    """
    pattern = _KV_PATTERN if ignore_spaces else _KV_PATTERN_STRICT
    m = pattern.match(source)
    if not m:
        raise ValueError('Syntax error in parsing environmental variable: {!r}'.
                         format(source))
    return m.groups()


def parse_env_file(path):
    """
    Parse environmental variables from a file.

    The spaces at the front and tail of each line, and surrounding "=", will
    be ignored.  Also, lines starting with "#" will be treated as a comment.

    Args:
        path (str): Path of the file.

    Returns:
        dict[str, str]: The parsed env dict.
    """
    ret = {}
    with codecs.open(path, 'rb', 'utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                name, val = parse_env(line)
                ret[name] = val
    return ret
