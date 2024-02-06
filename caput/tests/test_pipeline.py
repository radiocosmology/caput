"""Test running the caput.pipeline."""

from caput.tests import conftest


def test_pipeline():
    """Test running a very simple pipeline."""
    result = conftest.run_pipeline()
    print(result.output)
    assert result.exit_code == 0


def test_pipeline_multiple_outputs():
    """Test running a very simple pipeline with a multi-output task."""
    result = conftest.run_pipeline(configstr=conftest.multi_eggs_pipeline_conf)
    print(result.output)
    assert result.exit_code == 0
