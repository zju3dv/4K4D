function calibrate() {
    # Store default arguments
    : ${video:=1}
    : ${background1f_video:=3}
    : ${ground1f_video:=2}
    : ${human1f_video:=4}
    : ${root:=5}
    colmap=${colmap:-${6:-colmap}}
    phdeform=${phdeform:-${6:-/home/xuzhen/code/phdeform}}
    easymocap=${easymocap:-${6:-/home/xuzhen/code/easymocap-public}}

    # Let user know about the argument used
    echo "Possible arguments:"
    echo "video (abs): ${video}"
    echo "background1f_video (rlt): ${background1f_video}"
    echo "ground1f_video (rlt): ${ground1f_video}"
    echo "human1f_video (rlt): ${human1f_video}"
    echo "root (abs): ${root}"

    echo "Executables & repositories"
    echo "colmap: ${colmap}"
    echo "phdeform: ${phdeform}"
    echo "easymocap: ${easymocap}"

    # Since extracting videos is typically slow, will skip accordingly
    # todo: remove duplicated code
    old_cwd=$PWD
    cd $phdeform
    if [ -d "${root}/background1f" ]; then
        echo "${root}/background1f exists. Skipping extracting background1f from video"
    else
        python scripts/zjumocap/extract_videos.py ${video}/${background1f_video} ${root}/background1f --start 0 --end 1
    fi
    if [ -d "${root}/ground1f" ]; then
        echo "${root}/ground1f exists. Skipping extracting background1f from video"
    else
        python scripts/zjumocap/extract_videos.py ${video}/${ground1f_video} ${root}/ground1f --start 0 --end 1
    fi
    if [ -d "${root}/human1f" ]; then
        echo "${root}/human1f exists. Skipping extracting background1f from video"
    else
        python scripts/zjumocap/extract_videos.py ${video}/${human1f_video} ${root}/human1f --start 0 --end 1
    fi

    # Reconstruction using colmap
    cd $easymocap
    colmap_human1f_target=${root}/colmap-human1f/sparse/0/cameras.bin
    if [ -f "$colmap_human1f_target" ]; then
        echo "$colmap_human1f_target exists. Skipping colmap reconstruction"
    else
        python3 apps/calibration/calib_dense_by_colmap.py ${root}/human1f ${root}/colmap-human1f --share_camera --colmap ${colmap}
    fi
    # Detect chessboard
    ground1f_chessboard_target=${root}/ground1f/output
    if [ -d "$ground1f_chessboard_target" ]; then
        echo "$ground1f_chessboard_target exists. Skipping detecting chessboard"
    else
        python3 apps/calibration/detect_chessboard.py ${root}/ground1f --out ${root}/ground1f/output --pattern 11,8 --grid 0.06
    fi
    # Align results of chessboard detection and colmap
    colmap_align_target=${root}/colmap-align/intri.yml
    if [ -f "$colmap_align_target" ]; then
        echo "$colmap_align_target exists. Skipping aligning colmap and chessboard"
    else
        python3 apps/calibration/align_colmap_ground.py ${root}/colmap-human1f/sparse/0 ${root}/colmap-align --plane_by_chessboard ${root}/ground1f
    fi

    # Always save calibration check results
    python3 apps/calibration/check_calib.py ${root}/ground1f --mode cube --out ${root}/colmap-align --write
    python3 apps/calibration/check_calib.py ${root}/ground1f --mode match --out ${root}/colmap-align --write --annot chessboard

    cd $old_cwd
}
