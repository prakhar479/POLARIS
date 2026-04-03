ARG BASE_IMAGE=vcnk4v/suave-polaris:snapshot-20260404
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

USER root

ARG POLARIS_HOME=/opt/POLARIS

RUN set -euo pipefail; \
    src_launch="/home/kasm-user/suave_ws/src/suave/suave_missions/launch/suave_polaris.launch.py"; \
    install_launch="/home/kasm-user/suave_ws/install/suave_missions/share/suave_missions/launch/suave_polaris.launch.py"; \
    for f in "$src_launch" "$install_launch"; do \
      if [ -f "$f" ]; then \
        sed -i "s/'task_bridge': 'False'/'task_bridge': 'True'/g" "$f"; \
      fi; \
    done

WORKDIR ${POLARIS_HOME}

COPY pyproject.toml setup.py README.md ${POLARIS_HOME}/
COPY polaris ${POLARIS_HOME}/polaris
COPY config ${POLARIS_HOME}/config

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -e ${POLARIS_HOME}

RUN chown -R kasm-user:kasm-user ${POLARIS_HOME}

USER kasm-user
WORKDIR ${POLARIS_HOME}

ENV PYTHONUNBUFFERED=1
ENV POLARIS_CONFIG=/opt/POLARIS/config/suave.yaml

CMD ["/bin/bash"]
