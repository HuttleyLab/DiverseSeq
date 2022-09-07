# Overview

This repository contains code for a software development project that includes a command-line interface with a project structure that follows [best practice for Python projects](https://hynek.me/articles/testing-packaging/). The config file (`pyproject.toml`) shows the minimal information necessary to define a command-line tool where the build is controlled using [flit](https://pypi.org/project/flit). It also includes example configuration for dependencies, code-style tools ([black](https://pypi.org/project/black), [isort](https://pypi.org/project/isort)) and setting [pytest](https://pypi.org/project/pytest) options. 

Edit `pyproject.toml` file to fit your project.

If you want the command-line interface, edit section `project.scripts` (line 31). (If you don't want it, just delete that section.)

The `src/myproject/cli.py` presents an example of a command line application that uses `SciTrack` for logging, and `click` for specifying the interface.

Once you have edited this toml (making sure you've added it and any others you need into your project repository), you can do a "developer" install as

```
$ flit install -s --python `which python`
```

When it's correctly installed, you can run the following

```
$ myproject demo-echo "blah blah"
blah blah
```
where the output will be blue.

## Testing

I've included a test file that uses the `click` test runner. I have also included a `noxfile.py`

In the top-level directory, the following commands will run the tests

```
$ pytest
```

or

```
$ nox -s test-3.10
```

which will also generate a xml formatted coverage report. This is useful within a GitHub action for upload to codecov.

## Test coverage

I have also included a `nox` subcommand for producing a code coverage report.

```
$ nox -s testcov
```

Which will produce a coverage report in html format in the top level directory.

## Hack away 🤔 and enjoy 🎉!

 
