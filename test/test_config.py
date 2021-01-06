"""Test suite for configuration expansion."""
from click.testing import CliRunner
import itertools
from tinker.main import main


def test_vanilla():
    """Test "vanilla" configuration expansion."""
    runner = CliRunner()
    result = runner.invoke(main, ["examples/config/show_config.py", "-c", "examples/config/vanilla.yaml"])

    # The operation should succeed.
    assert result.exit_code == 0

    # The expected config object should appear.
    lines = result.output.strip().split("\n")
    assert len(lines) == 1
    assert lines[0] == str({
        "foo": 3,
        "bar": 4,
        "baz": 5,
    })


def test_iterate():
    """Test basic `iterate` directive."""
    runner = CliRunner()
    result = runner.invoke(main, ["examples/config/show_config.py", "-c", "examples/config/iterate.yaml"])

    # The operation should succeed.
    assert result.exit_code == 0

    # There should be six configurations.
    lines = result.output.strip().split("\n")
    assert len(lines) == 6

    # The expected config objects should appear in order.
    for i, (bar, baz) in enumerate(itertools.product([4, 5, 6], [7, 8])):
        expected = {
            "foo": 3,
            "bar": bar,
            "baz": baz,
        }

        assert lines[i] == str(expected)


def test_iterate_nested():
    """Test nested `iterate` directive."""
    runner = CliRunner()
    result = runner.invoke(main, ["examples/config/show_config.py", "-c", "examples/config/iterate_nested.yaml"])

    print(result.output)

    # The operation should succeed.
    assert result.exit_code == 0

    # There should be eight configurations.
    lines = result.output.strip().split("\n")
    assert len(lines) == 8

    # The expected config objects should appear in order.
    for i, (bar, baz) in enumerate(itertools.product([4, {"a": 10}, {"a": 12}, 6], [7, 8])):
        expected = {
            "foo": 3,
            "bar": bar,
            "baz": baz,
        }

        assert lines[i] == str(expected)
