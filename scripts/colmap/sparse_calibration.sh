function calibration() {
    # Run wild + multiple sparse setting calibration setting in easymocap
    # See https://chingswy.github.io/easymocap-public-doc/quickstart/calibration/wild-sparse.html

    # Some default settings, workspace, colmap, etc.
    emocap_root="${EASYMOCAP_DIR:-~/EasyMocap}"
    colmap_root="${COLMAP_DIR:-/usr/local/bin/colmap}"

    # Some calibration parameters
    scene_root="${1:-~/EasyMocap/data}"
    pattern="${2:-9,6}"
    grid="${3:-0.0250}"
    step="${4:-4}"

    # NOTE: you need to rename the .MOV file to scan.MOV before you run this script
    # Convert .MOV to .mp4
    ffmpeg -i $scene_root/background1f/scan.MOV -c:v libx264 -crf 18 -c:a aac -b:a 128K -movflags +faststart $scene_root/background1f/scan.mp4

    # Record the current working directory
    cwd=$PWD

    # Go to the EasyMocap directory
    cd $emocap_root
    # Run the calibration script
    echo "Running calibration..."

    # 1. Detect the chessboard
    echo "Detect the chessboard..."
    python3 apps/calibration/detect_chessboard.py $scene_root/ground1f --out $scene_root/ground1f/output --pattern $pattern --grid $grid

    # 2. Calibrate by colmap
    echo "Calibrate by colmap..."
    python3 apps/calibration/calib_static_dynamic_by_colmap.py $scene_root/background1f $scene_root/colmap --colmap $colmap_root --step $step

    # 3. Align to the chessboard
    # Remember to add `-X` command line argument when you are using `ssh`, this procedure will invoke `open3d`
    echo "Align to the chessboard..."
    python3 apps/calibration/align_colmap_ground.py $scene_root/colmap/sparse/0 $scene_root/colmap/align --plane_by_chessboard $scene_root/ground1f --prefix static/images/

    # Finally, go back to the original working directory
    echo "Calibration done!"
    cd $cwd
}

# source scripts/colmap/sparse_calibration.sh && CUDA_VISIBLE_DEVICES=0, calibration
