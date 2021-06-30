""" Delta dictionary
"""


class DeltaDict(dict):
    """ Implements basic properties of the delta dictionary.
    """

    def __add__(self, other):
        for k, v in other.items():
            if k not in self:
                self[k] = v
            else:
                self[k] += v

        return self

    def __neg__(self):
        return DeltaDict({k: -v for k, v in self.items()})
