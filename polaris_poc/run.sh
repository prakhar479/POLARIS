#!/bin/zsh

# ==============================================================================
# Launch Script for Polaris Services
#
# Description:
# This script automates the process of starting the NATS server and all
# required Polaris components. It opens a new terminal for each process.
#
# Shell:
# This script is configured to use zsh.
#
# Terminal Emulator:
# This script uses `gnome-terminal`. If you use a different terminal,
# you may need to adjust the `TERMINAL_CMD` variable.
# Examples:
#   - For Konsole (KDE): TERMINAL_CMD="konsole -e"
#   - For XFCE Terminal: TERMINAL_CMD="xfce4-terminal -e"
#   - For Xterm:         TERMINAL_CMD="xterm -e"
# ==============================================================================

# --- Configuration ---
# Auto-detect a suitable terminal emulator.
TERMINAL_EXEC=""
if command -v gnome-terminal &> /dev/null; then
    TERMINAL_EXEC="gnome-terminal"
elif command -v gnome-terminal- &> /dev/null; then
    TERMINAL_EXEC="gnome-terminal-"
elif command -v konsole &> /dev/null; then
    TERMINAL_EXEC="konsole"
elif command -v xfce4-terminal &> /dev/null; then
    TERMINAL_EXEC="xfce4-terminal"
elif command -v xterm &> /dev/null; then
    TERMINAL_EXEC="xterm"
else
    echo "Error: Could not find a supported terminal emulator (gnome-terminal, konsole, xfce4-terminal, xterm)." >&2
    echo "Please edit this script and set the 'TERMINAL_EXEC' variable manually to your terminal emulator." >&2
    exit 1
fi

# The command to activate the virtual environment.
ACTIVATE_ENV="source polaris/bin/activate"

# --- Main Script ---

# Check if the activation script exists
if [ ! -f "polaris/bin/activate" ]; then
    echo "Error: The activation script 'polaris/bin/activate' was not found."
    echo "Please run this script from the correct root directory."
    exit 1
fi

echo "🚀 Starting Polaris Services..."

# 1. Start the NATS server in a new terminal
echo "   -> Launching NATS Server..."
NATS_CMD_STRING="
  echo 'Activating environment and starting NATS server...';
  $ACTIVATE_ENV;
  ./bin/nats-server;
  echo 'NATS server process finished. Press Ctrl+C to close this terminal.';
  exec zsh"

case "$TERMINAL_EXEC" in
    "gnome-terminal" | "gnome-terminal-")
        $TERMINAL_EXEC --title="NATS Server" -- zsh -c "$NATS_CMD_STRING" &
        ;;
    "konsole")
        konsole --new-tab --title "NATS Server" -e zsh -c "$NATS_CMD_STRING" &
        ;;
    "xfce4-terminal")
        xfce4-terminal --title="NATS Server" --command="zsh -c '$NATS_CMD_STRING'" &
        ;;
    "xterm")
        xterm -T "NATS Server" -e "zsh -c '$NATS_CMD_STRING'" &
        ;;
esac

# Give the NATS server a moment to initialize before clients connect.
sleep 2

# 2. Define the list of components to start
components=("monitor" "kernel" "reasoner" "knowledge-base" "execution")

# 3. Loop through the components and start each one in a new terminal
for component in "${components[@]}"
do
  echo "   -> Launching Component: $component..."
  PYTHON_CMD="python3 src/scripts/start_component.py $component --plugin-dir extern"
  COMPONENT_CMD_STRING="
    echo 'Activating environment for component: $component...';
    $ACTIVATE_ENV;
    echo 'Starting component...';
    $PYTHON_CMD;
    echo 'Component process ($component) finished. Press Ctrl+C to close this terminal.';
    exec zsh"
  
  TITLE="Component: $component"

  case "$TERMINAL_EXEC" in
    "gnome-terminal" | "gnome-terminal-")
        $TERMINAL_EXEC --title="$TITLE" -- zsh -c "$COMPONENT_CMD_STRING" &
        ;;
    "konsole")
        konsole --new-tab --title "$TITLE" -e zsh -c "$COMPONENT_CMD_STRING" &
        ;;
    "xfce4-terminal")
        xfce4-terminal --title="$TITLE" --command="zsh -c '$COMPONENT_CMD_STRING'" &
        ;;
    "xterm")
        xterm -T "$TITLE" -e "zsh -c '$COMPONENT_CMD_STRING'" &
        ;;
  esac
  
  # Stagger the launch of components slightly
  sleep 0.5
done

echo ""
echo "✅ All services have been launched in separate terminal windows."

