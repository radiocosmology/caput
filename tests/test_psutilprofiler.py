"""Test running the caput.pipeline."""


def test_pipeline(run_pipeline):
    """Test profiling a very simple pipeline."""
    result = run_pipeline(["--psutil"])
    print(result.output)
    assert result.exit_code == 0
