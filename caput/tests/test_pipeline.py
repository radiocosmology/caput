"""Test running the caput.pipeline."""
from caput.tests import conftest


def test_pipeline():
    """Test running a very simple pipeline."""
    result = conftest.run_pipeline()
    print(result.output)
    assert result.exit_code == 0
