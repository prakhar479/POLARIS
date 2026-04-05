# SUAVE + Polaris Docker (Public Usage)

This folder contains a reproducible Docker setup for running SUAVE with Polaris as the adaptation manager.

## Option A: Use prebuilt image from Docker Hub

```bash
docker pull vcnk4v/suave-polaris-managed:latest

docker run -it --shm-size=512m \
  -p 6901:6901 \
  -p 9090:9090 \
  -e VNC_PW=password \
  --security-opt seccomp=unconfined \
  --name suave_polaris_managed \
  vcnk4v/suave-polaris-managed:latest
```

Inside container, run these in separate terminals:

```bash
# Terminal 1
sim_vehicle.py -L RATBeach -v ArduSub --model=JSON --console
```

```bash
# Terminal 2
source /opt/ros/humble/setup.bash
source /home/kasm-user/suave_ws/install/setup.bash
ros2 launch suave simulation.launch.py x:=-17.0 y:=2.0
```

```bash
# Terminal 3
source /opt/ros/humble/setup.bash
source /home/kasm-user/suave_ws/install/setup.bash
ros2 launch suave_missions mission.launch.py adaptation_manager:=polaris mission_type:=time_constrained_mission
```

```bash
# Terminal 4 (Polaris)
python -m polaris.cli --config config/suave.yaml --export-logs /opt/POLARIS/polaris.log
```

## Option B: Build locally from this repository

From repository root:

```bash
docker build \
  -f examples/managed-system/suave/Dockerfile \
  --build-arg BASE_IMAGE=vcnk4v/suave-polaris:snapshot-20260406 \
  -t vcnk4v/suave-polaris-managed:v1 \
  .
```

Run:

```bash
docker run -it --shm-size=512m \
  -p 6901:6901 \
  -p 9090:9090 \
  -e VNC_PW=password \
  --security-opt seccomp=unconfined \
  --name suave_polaris_managed \
  vcnk4v/suave-polaris-managed:v1
```

## Publish to Docker Hub (for maintainers)

### 1) Push your base snapshot image

```bash
docker tag suave_polaris:snapshot-20260406 vcnk4v/suave-polaris:snapshot-20260406
docker push vcnk4v/suave-polaris:snapshot-20260406
```

Optional rolling tag:

```bash
docker tag suave_polaris:snapshot-20260406 vcnk4v/suave-polaris:latest
docker push vcnk4v/suave-polaris:latest
```

### 2) Build and push managed image

```bash
docker build \
  -f examples/managed-system/suave/Dockerfile \
  --build-arg BASE_IMAGE=vcnk4v/suave-polaris:snapshot-20260406 \
  -t vcnk4v/suave-polaris-managed:v1 \
  -t vcnk4v/suave-polaris-managed:latest \
  .

docker push vcnk4v/suave-polaris-managed:v1
docker push vcnk4v/suave-polaris-managed:latest
```

## Verify SUAVE interfaces

After launching Terminals 1–3, required interfaces should exist:

- services: `/task/request`, `/task/cancel`, `/*/change_mode`
- topics: `/diagnostics`, `/pipeline/detected`

Quick check:

```bash
source /opt/ros/humble/setup.bash
source /home/kasm-user/suave_ws/install/setup.bash
ros2 daemon stop || true
ros2 daemon start
sleep 2
ros2 service list | egrep 'task/request|task/cancel|/f_.*change_mode'
ros2 topic list | egrep '^/diagnostics$|^/pipeline/detected$'
```
