LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")

test:
	pytest tests/


quality_checks:
				isort .
				black .

run: test quality_checks
	echo  test completed!