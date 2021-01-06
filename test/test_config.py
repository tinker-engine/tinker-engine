"""Test suite for configuration expansion."""
from click.testing import CliRunner
import itertools
from tinker.main import main


def test_iterate():
    """Test basic `iterate` directive."""
    runner = CliRunner()
    result = runner.invoke(main, ["examples/config/show_config.py", "-c", "examples/config/iterate.yaml"])

    # The operation should succeed.
    assert result.exit_code == 0

    # The expected config objects should appear in order.
    lines = result.output.split("\n")
    for i, (bar, baz) in enumerate(itertools.product([4, 5, 6], [7, 8])):
        expected = {
            "foo": 3,
            "bar": bar,
            "baz": baz,
        }

        assert lines[i] == str(expected)
