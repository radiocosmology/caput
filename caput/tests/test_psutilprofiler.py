"""Test running the caput.pipeline."""

from caput.tests import conftest


def test_pipeline():
    """Test profiling a very simple pipeline."""
    result = conftest.run_pipeline(["--psutil"])
    print(result.output)
    assert result.exit_code == 0
