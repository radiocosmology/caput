# Skyfield Data Cache

This contains a cache of the data that skyfield needs to function. In theory
skyfield can download this on demand, but in practice that seems to be flaky
when running on clusters or on Travis.