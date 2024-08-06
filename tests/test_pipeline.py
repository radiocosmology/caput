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


def test_manual_pipeline_interactive(get_pipeline):
    """Test running a pipeline with interactive mode enabled.

    Manually walk through each step of the config and check that
    tasks are exectued as expected. Breakpoints should have no
    effect.
    """

    interactive_eggs_pipeline_conf = """
---
pipeline:
  interactive: true
  enable_breakpoints: true
  tasks:
    - type: tests.conftest.PrintEggs
      params: eggs_params
    - type: tests.conftest.GetEggs
      params: eggs_params
      out: egg
    - type: tests.conftest.CookEggs
      params: cook_params
      in: egg
eggs_params:
  eggs: ['green', 'duck', 'ostrich']
  breakpoint: true
cook_params:
  style: 'fried'
"""

    manager = get_pipeline(configstr=interactive_eggs_pipeline_conf)
    # Get the pipeline runner generator
    runner = manager.runner()

    # Skip through setup state
    for _ in range(4):
        next(runner)

    # Check that all tasks are in the expected state
    assert manager._task_idx == 0
    for task in manager.tasks:
        assert task._pipeline_state == "next"

    # Run the pipeline to the end
    while True:
        try:
            next(runner)
        except StopIteration:
            break

    # Verify that the pipeline is complete
    assert all(task is None for task in manager.tasks)


def test_manual_pipeline(get_pipeline):
    """Test running a pipeline with interactive mode and breakpoints disabled."""

    breakpoint_eggs_pipeline_conf = """
---
pipeline:
  interactive: false
  enable_breakpoints: false
  tasks:
    - type: tests.conftest.PrintEggs
      params: eggs_params
    - type: tests.conftest.GetEggs
      params: eggs_params
      out: egg
    - type: tests.conftest.CookEggs
      params: cook_params
      in: egg
eggs_params:
  eggs: ['green']
  breakpoint: true
cook_params:
  style: 'fried'
"""

    manager = get_pipeline(configstr=breakpoint_eggs_pipeline_conf)

    # Run the pipeline
    manager.run()

    # Verify that the pipeline is complete
    assert all(task is None for task in manager.tasks)
