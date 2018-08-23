import json
import unittest

from mlstorage_client.utils import Tokens, parse_config, parse_tags


class TokensTestCase(unittest.TestCase):

    def test_config_value_token(self):
        t = Tokens.config_value
        g = lambda a, b: \
            self.assertEqual(a, t.parseString(b, parseAll=True).asList()[0])
        g(123, '123')
        g(123, ' 123 ')
        g(-123, '-123')
        g(-123, ' -123 ')
        g(123.0, '123.0')
        g(123.0, ' 123.0 ')
        g('', '')
        for true_literal in ('true', 'True', 'yes', 'on'):
            g(True, true_literal)
            g(True, ' {} '.format(true_literal))
        for false_literal in ('false', 'False', 'no', 'off'):
            g(False, false_literal)
            g(False, ' {} '.format(false_literal))
        for unquoted_string in ('123 456', 'true false', 'hello "world"'):
            g(unquoted_string, unquoted_string)
            g(unquoted_string, ' {} '.format(unquoted_string))
            quoted_string = json.dumps(unquoted_string)
            g(unquoted_string, quoted_string)
        g('hello, "world"', '"hello, \\"world\\""')
        g('hello, "world"', ' "hello, \\"world\\"" ')

    def test_key_value_pair_list_token(self):
        t = Tokens.key_value_pair_list
        g = lambda a, b: \
            self.assertListEqual(a, t.parseString(b, parseAll=True).asList())
        g([], '')
        g([('a', 1)], 'a=1')
        g([('a', 1), ('b', 123.0), ('c', True), ('d', 'true false'),
           ('e', 'hello "world"'), ('f', 'hello, "world"')],
          'a=1,b=123.0,c=true,d=true false,e=hello "world",'
          'f="hello, \\"world\\""')
        g([('a', 1), ('b', 123.0), ('c', True), ('d', 'true false'),
           ('e', 'hello "world"'), ('f', 'hello, "world"')],
          'a = 1 , b = 123.0 , c = true , d = true false , '
          'e = hello "world" , f = "hello, \\"world\\"" ')

    def test_tag_list_token(self):
        t = Tokens.tag_list
        g = lambda a, b: \
            self.assertListEqual(a, t.parseString(b, parseAll=True).asList())
        g([], '')
        g(['a'], 'a')
        g(['a', 'b', 'c d', 'hello "world"', 'hello, "world"'],
          'a, "b", c d, hello "world", "hello, \\"world\\""')


class ParseConfigTestCase(unittest.TestCase):

    def test_parse_config(self):
        self.assertDictEqual(
            {'a': 1, 'b': 123.0, 'c': True, 'd': 'true false',
             'e': 'hello "world"', 'f': 'hello, "world"'},
            parse_config('a=1,b=123.0,c=true,d=true false,'
                         'e=hello "world",f="hello, \\"world\\""')
        )


class ParseTagsTestCaes(unittest.TestCase):

    def test_parse_tags(self):
        self.assertListEqual(
            ['a', 'b', 'c d', 'hello "world"', 'hello, "world"'],
            parse_tags('a, "b", c d, hello "world", "hello, \\"world\\"", ,')
        )


if __name__ == '__main__':
    unittest.main()
