## Start NATS server

```
./bin/nats-server
```

## Start Monitor 

```
python3 src/scripts/start_component.py monitor --plugin-dir extern
```

## Start Kernel

(From src directory)

```
python3 -m polaris.kernel.kernel
```

## Start Executor

```
python3 src/scripts/start_component.py execution --plugin-dir extern
```