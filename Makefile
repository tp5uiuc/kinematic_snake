black:
	@black --version
	@black kinematic_snake examples tests

black_check:
	@black --version
	@black --check kinematic_snake examples tests

isort:
	@isort --version
	@isort --recursive .

isort_check:
	@isort --version
	@isort --recursive --check-only

flake8:
	@flake8 --version
	@flake8 kinematic_snake examples tests

clean_notebooks:
    # This finds Ipython jupyter notebooks in the code
    # base and cleans only its output results. This
    # results in
	@jupyter nbconvert --version
	@find . -maxdepth 3 -name '*.ipynb'\
		| while read -r src; do jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$$src"; done

pylint:
	@pylint --version
	@find . -maxdepth 3 -name '*.py'\
		| while read -r src; do pylint -rn "$$src"; done

all:black flake8
ci:black_check flake8
