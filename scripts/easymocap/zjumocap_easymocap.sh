function easymocap() {
    # run idr pipeline of easymocap
    data_root="${1:-/nas/home/xuzhen/datasets/dynacap/vlad}"
    outdir="${2:-$data_root/easymocap}"
    mocap_modes=("$@")
    if [[ ${#mocap_modes} -gt 2 ]]; then
        mocap_modes=${mocap_modes:2:$((${#mocap_modes} - 2))}
    else
        mocap_modes=(smpl-3d smplh-3d vposer-3d vposerh-3d)
    fi

    echo "will use mocap modes: $mocap_modes"

    easymocap_dir=${EASYMOCAP_DIR:-/home/xuzhen/code/easymocap-public}
    # openpose_dir=${OPENPOSE_DIR:-/nas/users/xuzhen/code/openpose}
    echo $openpose_dir
    cwd=$PWD
    # mocap_mode="${2:-smplh-3d}"
    export PYOPENGL_PLATFORM=osmesa
    # export LD_LIBRARY_PATH=$openpose_dir/build/src/openpose:$openpose_dir/build/caffe/lib:$LD_LIBRARY_PATH

    echo "prepare easymocap data_root"
    mkdir -p $outdir
    cd $outdir
    ln -sfn $data_root/intri.yml
    ln -sfn $data_root/extri.yml
    ln -sfn $data_root/images

    echo "running easymocap pipeline"
    cd $easymocap_dir

    echo "extracting keypoints"
    # when using openpose, add --hand to run hand keypoints too
    # python3 apps/preprocess/extract_keypoints.py "$outdir" --mode openpose --openpose "$openpose_dir" --hand
    python3 apps/preprocess/extract_keypoints.py "$outdir" --mode mp-holistic

    echo "checking camera pose and triangulations"
    python3 apps/calibration/check_calib.py $outdir --out $outdir --mode human --write --hand

    for mode in $mocap_modes; do
        echo "running mocap: $mode"
        # check here for vposer & pyrender installation: /home/xuzhen/easymocap-public/scripts/install
        # this will also write smplfull: *.json, with all angle-axis representation
        python3 apps/demo/mocap.py "$outdir" --mode $mode --ranges 0 $(ls $outdir/images/00 | wc -l) 1
        # python3 apps/demo/mocap.py "$outdir" --mode $mode

        # check fitting results as mesh file
        echo "writting vertices"
        postfix=output-smpl-3d
        python3 apps/postprocess/write_vertices.py "$outdir/output-$postfix/smpl" "$outdir/output-$postfix/mesh" --cfg_model "$outdir/output-$postfix/cfg_model.yml" --mode mesh
        python3 apps/postprocess/write_vertices.py "$outdir/output-$postfix/smpl" "$outdir/output-$postfix/vertices" --cfg_model "$outdir/output-$postfix/cfg_model.yml" --mode vertices
        # python3 apps/postprocess/write_vertices.py "$outdir/output-$mode/smpl" "$outdir/output-$mode/mesh" --cfg_model "$outdir/output-$mode/cfg_model.yml" --mode mesh
        # python3 apps/postprocess/write_vertices.py "$outdir/output-$mode/smpl" "$outdir/output-$mode/vertices" --cfg_model "$outdir/output-$mode/cfg_model.yml" --mode vertices
    done

    cd $cwd
}

# source scripts/zjumocap/zjumocap_easymocap.sh && CUDA_VISIBLE_DEVICES=7, easymocap

# source scripts/zjumocap/zjumocap_easymocap.sh
# for exp in my_313 my_315 my_377 my_386 my_387 my_390 my_392 my_393 my_394; do easymocap "/nas/users/xuzhen/datadirsets/zju-mocap/$exp"; done
# source script/zjumocap/zjumocap_easymocap.sh
# for exp in 00 01 02; do easymocap /nas/users/xuzhen/datadirsets/mobile-stage/dance/instance/$exp; done
