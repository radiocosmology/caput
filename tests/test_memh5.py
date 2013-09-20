"""Unit tests for the memh5 module."""

import unittest

from caput import memh5

class TestRODict(unittest.TestCase):

    def test_everything(self):
        a = {'a' : 5}
        a = memh5.ro_dict(a)
        self.assertEqual(a['a'], 5)
        self.assertEqual(a.keys(), ['a'])
        # Convoluded test to make sure you can't write to it.
        try: a['b'] = 6
        except TypeError: correct = True
        else: correct = False
        self.assertTrue(correct)

class TestGroup(unittest.TestCase):

    def test_nested(self):
        pass


if __name__ == '__main__':
    unittest.main()
