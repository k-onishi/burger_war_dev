function my_docker_exec() {
  local DOCKER_USER="$1"
  local CONTAINER="$2"
  local COMMANDS="$3"
  local WORKING_DIR="/home/${DOCKER_USER}"

  if [ -z "${COMMANDS}" ]; then
    COMMANDS=/bin/bash
  fi

  docker exec -it -u ${DOCKER_USER} -w ${WORKING_DIR} ${CONTAINER} ${COMMANDS}
}

function de() {
  COMMANDS="$*"
  my_docker_exec ubuntu burger_war_docker "${COMMANDS}"
}
