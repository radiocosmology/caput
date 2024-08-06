"""Test running the caput.pipeline."""


def test_pipeline(run_pipeline):
    """Test running a very simple pipeline."""
    result = run_pipeline()
    print(result.output)
    assert result.exit_code == 0


def test_pipeline_multiple_outputs(run_pipeline):
    """Test running a very simple pipeline with a multi-output task."""

    multi_eggs_pipeline_conf = """
---
pipeline:
  tasks:
    - type: tests.conftest.GetEggs
      params: eggs_params
      out: [color, egg]
    - type: tests.conftest.CookEggs
      params: cook_params
      in: egg
    - type: tests.conftest.CookEggs
      params: cook_params
      in: color
eggs_params:
  eggs: [['green', 'duck'], ['blue', 'ostrich']]
cook_params:
  style: 'fried'
"""

    result = run_pipeline(configstr=multi_eggs_pipeline_conf)
    print(result.output)
    assert result.exit_code == 0
