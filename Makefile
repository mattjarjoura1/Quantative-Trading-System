.PHONY: install run clean

install:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

run:
	@echo "Activating venv — run 'deactivate' to exit"
	@bash -c "source venv/bin/activate && exec bash"

clean:
	rm -rf venv
