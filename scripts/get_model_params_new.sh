#!/bin/bash

# Check if directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <output_directory>"
    echo "Example: bash $0 ./model_params"
    exit 1
fi

# Check if curl is available
if ! command -v curl &> /dev/null; then
    echo "Error: curl is required but not installed. Please install curl first." >&2
    exit 1
fi

# Create output directory
mkdir -p "$1"

# Base URL
BASE_URL="https://files.ipd.uw.edu/pub/ligandmpnn"

# Function to download a file
download_file() {
    local url="$1"
    local output="$2"
    echo "Downloading: $(basename "$output")"
    curl -s -L -o "$output" "$url" || echo "Error downloading: $url" >&2
}

# Original ProteinMPNN weights
for version in 002 010 020 030; do
    download_file "$BASE_URL/proteinmpnn_v_48_${version}.pt" "$1/proteinmpnn_v_48_${version}.pt"
done

# LigandMPNN with num_edges=32; atom_context_num=25
for version in 005 010 020 030; do
    download_file "$BASE_URL/ligandmpnn_v_32_${version}_25.pt" "$1/ligandmpnn_v_32_${version}_25.pt"
done

# Per residue label membrane ProteinMPNN
download_file "$BASE_URL/per_residue_label_membrane_mpnn_v_48_020.pt" \
    "$1/per_residue_label_membrane_mpnn_v_48_020.pt"

# Global label membrane ProteinMPNN
download_file "$BASE_URL/global_label_membrane_mpnn_v_48_020.pt" \
    "$1/global_label_membrane_mpnn_v_48_020.pt"

# SolubleMPNN
for version in 002 010 020 030; do
    download_file "$BASE_URL/solublempnn_v_48_${version}.pt" "$1/solublempnn_v_48_${version}.pt"
done

echo "Download completed successfully!"
