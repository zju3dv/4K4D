function easymocap_from_static_recon() {
    # run idr pipeline of easymocap
    default_data_root="/nas/dataset/static_recon/xwzhou/A_precise"
    data="${1:-$default_data_root}"
    easydir="${2:-easymocap}"
    camera_id="${3:-VID_12}"
    easydata="$data/$easydir"
    easymocap_public_dir="/home/xuzhen/easymocap-public"
    cwd=$PWD
    openpose="/nas/users/xuzhen/code/openpose"
    export PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=$openpose/build/src/openpose:$openpose/build/caffe/lib:$LD_LIBRARY_PATH

    # echo "prepareing easymocap data..."
    cd /home/xuzhen/dngp

    echo "converting camera parameters..."
    python scripts/read_colmap.py "$data" sfm_ws_scaled $camera_id --format .bin --output "$easydir"

    echo "copying images..."
    python scripts/img2easymocap.py --data_root "$data" --input colmap_images_undist --subs $camera_id --output "$easydir"

    echo "running easymocap pipeline..."
    cd $easymocap_public_dir

    echo "extracting keypoints..."
    # when using openpose, add --hand to run hand keypoints too
    # python3 apps/preprocess/extract_keypoints.py "$easydata" --mode openpose --openpose "$openpose" --hand
    python3 apps/preprocess/extract_keypoints.py "$easydata" --mode mp-holistic

    echo "checking camera pose and triangulations..."
    python3 apps/calibration/check_calib.py $easydata --out $easydata --mode human --write --hand

    echo "running mocap..."
    # check here for vposer & pyrender installation: /home/xuzhen/easymocap-public/scripts/install
    # this will also write smplfull: *.json, with all angle-axis representation
    python3 apps/demo/mocap.py "$easydata" --mode static --range 0 1 1

    echo "writting mesh vertices..."
    python3 apps/postprocess/write_vertices.py "$easydata/output-static/smpl" "$easydata/output-static/mesh" --cfg_model "$easydata/output-static/cfg_model.yml" --mode mesh
    python3 apps/postprocess/write_vertices.py "$easydata/output-static/smpl" "$easydata/output-static/vertices" --cfg_model "$easydata/output-static/cfg_model.yml" --mode vertices

    cd $cwd
}

# source scripts/idr_easymocap.sh && CUDA_VISIBLE_DEVICES=7, easymocap_from_static_recon "/nas/dataset/static_recon/xwzhou/pose1_precise" & CUDA_VISIBLE_DEVICES=6, easymocap_from_static_recon "/nas/dataset/static_recon/xwzhou/pose2_precise" &
function easymocap_only() {
    # run idr pipeline of easymocap
    default_data_root="/nas/dataset/static_recon/xwzhou/A_precise"
    data="${1:-$default_data_root}"
    easydir="${2:-easymocap-zhen}"
    easydata="$data/$easydir"
    easymocap_public_dir="/home/xuzhen/easymocap-public"
    cwd=$PWD
    openpose="/nas/users/xuzhen/code/openpose"
    ext="${3:-.jpg}"
    export PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=$openpose/build/src/openpose:$openpose/build/caffe/lib:$LD_LIBRARY_PATH

    echo "running easymocap pipeline..."
    cd $easymocap_public_dir

    echo "extracting keypoints..."
    # when using openpose, add --hand to run hand keypoints too
    # python3 apps/preprocess/extract_keypoints.py "$easydata" --mode openpose --openpose "$openpose" --hand
    python3 apps/preprocess/extract_keypoints.py "$easydata" --mode mp-holistic --ext $ext

    echo "checking camera pose and triangulations..."
    python3 apps/calibration/check_calib.py $easydata --out $easydata --mode human --write --hand

    echo "running mocap..."
    # check here for vposer & pyrender installation: /home/xuzhen/easymocap-public/scripts/install
    # this will also write smplfull: *.json, with all angle-axis representation
    python3 apps/demo/mocap.py "$easydata" --mode static --range 0 1 1

    echo "writting mesh vertices..."
    python3 apps/postprocess/write_vertices.py "$easydata/output-static/smpl" "$easydata/output-static/mesh" --cfg_model "$easydata/output-static/cfg_model.yml" --mode mesh
    python3 apps/postprocess/write_vertices.py "$easydata/output-static/smpl" "$easydata/output-static/vertices" --cfg_model "$easydata/output-static/cfg_model.yml" --mode vertices

    cd $cwd
}
