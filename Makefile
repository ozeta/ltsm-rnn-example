# run python test
.PHONY: test
test:
	pytest -s lambda/test/test.py  --log-cli-level debug --log-file=out.log