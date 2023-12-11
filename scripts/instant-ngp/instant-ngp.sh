function instant-ngp() {
    # run instant-ngp pipeline for camera parameters optimization
    cwd=$PWD
    data_dir="${1:-/nas/datasets/mgtv_nvs/mocap_mgtv_val/F1_06_000000}"
    image="${2:-000000.jpg}"
    image_dir="${3:-images}"
    mask_dir="${4:-bgmt}"
    ngp_dir="${5:-ngp}"
    intri_file="${6:-intri.yml}"
    extri_file="${7:-extri.yml}"

    INSTANT_NGP_DIR="${INSTANT_NGP_DIR:-${codespace}/3rdparty/instant-ngp}"
    RUN_INSTANT_NGP="${RUN_INSTANT_NGP:-false}"

    echo "converting easymocap camera parameters to colmap format"
    python3 scripts/instant-ngp/easymocap_to_colmap.py --image_dir $data_dir/$image_dir --mask_dir $data_dir/$mask_dir --output_dir $data_dir/$ngp_dir --intri $data_dir/$intri_file --extri $data_dir/$extri_file --image $image $EASYMOCAP2COLMAP_OPTS[@]

    echo "converting colmap camera parameters to nerf format (instant-ngp)"
    python3 scripts/instant-ngp/colmap_to_nerf.py --images $data_dir/$ngp_dir --text $data_dir/$ngp_dir --out_dir $data_dir/$ngp_dir $COLMAP2NERF_OPTS[@]

    cmd="python3 scripts/run.py --n_steps 3100 --mode nerf --save_snapshot --scene "$data_dir/$ngp_dir" --screenshot_transforms "$data_dir/$ngp_dir/eval/eval_transforms.json" $INSTANT_NGP_OPTS[@]" # now, with better mask, compositing with background is giving vdfq performance boost, around 1
    echo $cmd
    if [ $RUN_INSTANT_NGP = true ]; then
        cd $INSTANT_NGP_DIR
        echo "running instant-ngp pipeline"
        eval $cmd
    fi

    cd $cwd

}

function instant-ngp-eval() {
    # run instant-ngp pipeline for camera parameters optimization
    cwd=$PWD
    data_dir="${1:-/nas/datasets/mgtv_nvs/mocap_mgtv_val/F1_06_000000}"
    image="${2:-000000.jpg}"
    image_dir="${3:-images}"
    mask_dir="${4:-bgmt}"
    ngp_dir="${5:-ngp}"
    intri_file="${6:-intri.yml}"
    extri_file="${7:-extri.yml}"

    INSTANT_NGP_DIR="${INSTANT_NGP_DIR:-${codespace}/3rdparty/instant-ngp}"
    RUN_INSTANT_NGP="${RUN_INSTANT_NGP:-true}" # this time, defaults to run the thing

    echo "converting easymocap camera parameters to colmap format, this will place background images at target view path"
    python3 scripts/instant-ngp/easymocap_to_colmap.py --no_load_train --image_dir $data_dir/$image_dir --mask_dir $data_dir/$mask_dir --output_dir $data_dir/$ngp_dir --intri $data_dir/$intri_file --extri $data_dir/$extri_file --image $image $EASYMOCAP2COLMAP_OPTS[@]

    cmd="python3 scripts/run.py --mode nerf --load_snapshot --scene "$data_dir/$ngp_dir" --screenshot_transforms "$data_dir/$ngp_dir/eval/eval_transforms.json" $INSTANT_NGP_OPTS[@]" # now, with better mask, compositing with background is giving vdfq performance boost, around 1
    echo $cmd
    if [ $RUN_INSTANT_NGP = true ]; then
        cd $INSTANT_NGP_DIR
        echo "running instant-ngp evaluation pipeline"
        eval $cmd
    fi

    cd $cwd

}
