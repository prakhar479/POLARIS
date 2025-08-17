#!/bin/bash

# Robust NATS Server Installation Script
# Downloads, verifies, and installs NATS server executable

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# --- Configuration ---
NATS_VERSION="v2.11.6"  # Latest stable version confirmed from GitHub releases
BASE_URL="https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}"
INSTALL_DIR="${INSTALL_DIR:-./bin}"
TEMP_DIR="nats_temp_$$"  # Use PID for unique temp directory

# --- Color output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helper functions ---
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f nats_server.tar.gz nats_server.tar.gz.tmp
    rm -rf "${TEMP_DIR}"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# --- Prerequisite checks ---
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    command -v curl >/dev/null 2>&1 || missing_tools+=("curl")
    command -v unzip >/dev/null 2>&1 || missing_tools+=("unzip")
    command -v tar >/dev/null 2>&1 || missing_tools+=("tar")
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install them and try again."
        exit 1
    fi
}

# --- OS and Architecture Detection ---
detect_platform() {
    log_info "Detecting platform..."
    
    OS_NAME=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    
    # Normalize OS name
    case "$OS_NAME" in
        darwin) OS_NAME="darwin" ;;
        linux) OS_NAME="linux" ;;
        freebsd) OS_NAME="freebsd" ;;
        windows|mingw*|cygwin*|msys*) 
            OS_NAME="windows"
            log_warn "Windows detected. This script may need adjustments for Windows environments."
            ;;
        *)
            log_error "Unsupported operating system: $OS_NAME"
            exit 1
            ;;
    esac
    
    # Normalize architecture
    case "$ARCH" in
        x86_64|amd64) NATS_ARCH="amd64" ;;
        arm64|aarch64) NATS_ARCH="arm64" ;;
        armv6l) NATS_ARCH="arm6" ;;
        armv7l) NATS_ARCH="arm7" ;;
        i386|i686) NATS_ARCH="386" ;;
        *)
            log_error "Unsupported architecture: $ARCH"
            log_error "Supported architectures: amd64, arm64, arm6, arm7, 386"
            exit 1
            ;;
    esac
    
    log_success "Platform detected: ${OS_NAME}-${NATS_ARCH}"
}

# --- Download and verify ---
download_nats() {
    # NATS releases use tar.gz format, not zip
    local filename="nats-server-${NATS_VERSION}-${OS_NAME}-${NATS_ARCH}.tar.gz"
    local download_url="${BASE_URL}/${filename}"
    
    log_info "Downloading NATS Server from: $download_url"
    
    # Download with progress and retry
    if ! curl -L \
        --retry 3 \
        --retry-delay 2 \
        --connect-timeout 10 \
        --max-time 300 \
        --fail \
        --show-error \
        --progress-bar \
        "$download_url" \
        -o nats_server.tar.gz.tmp; then
        log_error "Download failed. Please check:"
        log_error "  1. Internet connection"
        log_error "  2. NATS version exists: $NATS_VERSION"
        log_error "  3. Platform support: ${OS_NAME}-${NATS_ARCH}"
        log_error "  4. Direct URL: $download_url"
        exit 1
    fi
    
    # Atomic move to avoid partial files
    mv nats_server.tar.gz.tmp nats_server.tar.gz
    
    # Verify the download is a valid tar.gz file
    if ! tar -tzf nats_server.tar.gz >/dev/null 2>&1; then
        log_error "Downloaded file is not a valid tar.gz archive"
        exit 1
    fi
    
    log_success "Download completed successfully"
}

# --- Extract and install ---
install_nats() {
    log_info "Extracting NATS server..."
    
    # Create temp directory
    mkdir -p "$TEMP_DIR"
    
    # Extract tar.gz archive
    log_info "Extracting TAR.GZ archive..."
    if ! tar -xzf nats_server.tar.gz -C "$TEMP_DIR"; then
        log_error "Failed to extract tar.gz archive"
        exit 1
    fi
    
    # Find the executable (handle different archive structures)
    local executable_path
    executable_path=$(find "$TEMP_DIR" -name "nats-server" -type f -executable 2>/dev/null | head -1)
    
    if [ -z "$executable_path" ]; then
        # Fallback: look for nats-server without executable bit check
        executable_path=$(find "$TEMP_DIR" -name "nats-server" -type f 2>/dev/null | head -1)
    fi
    
    if [ -z "$executable_path" ]; then
        log_error "Could not find 'nats-server' executable in the archive"
        log_error "Archive contents:"
        find "$TEMP_DIR" -type f
        exit 1
    fi
    
    log_info "Found executable: $executable_path"
    
    # Create install directory if it doesn't exist
    mkdir -p "$INSTALL_DIR"
    
    # Copy executable and set permissions
    cp "$executable_path" "$INSTALL_DIR/nats-server"
    chmod +x "$INSTALL_DIR/nats-server"
    
    log_success "NATS server installed to: $INSTALL_DIR/nats-server"
}

# --- Verify installation ---
verify_installation() {
    log_info "Verifying installation..."
    
    local nats_binary="$INSTALL_DIR/nats-server"
    
    if [ ! -f "$nats_binary" ]; then
        log_error "Installation verification failed: binary not found"
        exit 1
    fi
    
    if [ ! -x "$nats_binary" ]; then
        log_error "Installation verification failed: binary not executable"
        exit 1
    fi
    
    # Test the binary
    local version_output
    if version_output=$("$nats_binary" --version 2>&1); then
        log_success "Installation verified successfully!"
        echo "$version_output"
    else
        log_warn "Binary installed but version check failed"
        log_warn "This might be normal for some NATS versions"
    fi
    
    # Show file details
    log_info "Installation details:"
    ls -la "$nats_binary"
}

# --- Main execution ---
main() {
    log_info "Starting NATS Server installation..."
    log_info "Version: $NATS_VERSION"
    log_info "Install directory: $INSTALL_DIR"
    
    check_prerequisites
    detect_platform
    download_nats
    install_nats
    verify_installation
    
    log_success "NATS Server installation completed successfully!"
    log_info "You can now run: $INSTALL_DIR/nats-server"
    
    # Optionally add to PATH suggestion
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]] && [ "$INSTALL_DIR" != "./bin" ]; then
        log_info "To add to PATH, run: export PATH=\"$INSTALL_DIR:\$PATH\""
    fi
}

# --- Script execution ---
main "$@"