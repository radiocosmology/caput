import pickle

import pytest

from caput import config


### Test classes
class Person(config.Reader):
    name = config.Property(default="Bill", proptype=str)
    age = config.Property(default=26, proptype=float, key="ageinyears")


class PersonWithPet(Person):
    petname = config.Property(default="Molly", proptype=str)
    petage = 36


class ListTypeTests(Person):
    list_max_length = config.list_type(maxlength=2)
    list_exact_length = config.list_type(length=2)
    list_type = config.list_type(type_=int)


### Test data dict
testdict = {"name": "Richard", "ageinyears": 40, "petname": "Sooty"}


### Tests
def test_default_params():

    person1 = Person()

    assert person1.name == "Bill"
    assert person1.age == 26.0
    assert isinstance(person1.age, float)


def test_set_params():

    person = Person()
    person.name = "Mick"

    assert person.name == "Mick"


def test_read_config():

    person = Person()
    person.read_config(testdict)

    assert person.name == "Richard"
    assert person.age == 40.0


def test_inherit_read_config():

    person = PersonWithPet()
    person.read_config(testdict)

    assert person.name == "Richard"
    assert person.age == 40.0
    assert person.petname == "Sooty"


def test_pickle():

    person = PersonWithPet()
    person.read_config(testdict)
    person2 = pickle.loads(pickle.dumps(person))

    assert person2.name == "Richard"
    assert person2.age == 40.0
    assert person2.petname == "Sooty"


def test_list_type():

    lt = ListTypeTests()

    with pytest.raises(config.CaputConfigError):
        lt.read_config({"list_max_length": [1, 3, 4]})

    # Should work fine
    lt = ListTypeTests()
    lt.read_config({"list_max_length": [1, 2]})

    with pytest.raises(config.CaputConfigError):
        lt.read_config({"list_exact_length": [3]})

    # Work should fine
    lt = ListTypeTests()
    lt.read_config({"list_exact_length": [1, 2]})

    with pytest.raises(config.CaputConfigError):
        lt.read_config({"list_type": ["hello"]})

    # Work should fine
    lt = ListTypeTests()
    lt.read_config({"list_type": [1, 2]})
