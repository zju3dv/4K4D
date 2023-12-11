# add new images to colmap
import os
import argparse
from os.path import join

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.console_utils import log, run
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.colmap_utils import correct_colmap_scale
# fmt: on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/iphone/multiview_412')
    parser.add_argument('--colmap_db', default='colmap/colmap.db')
    parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"], help="camera model")
    parser.add_argument("--colmap_camera_params", default="", help="intrinsic parameters, depending on the chosen model.  Format: fx, fy, cx, cy, dist")
    parser.add_argument("--colmap_matcher", default="vocab_tree", choices=["exhaustive", "sequential", "spatial", "transitive", "vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
    parser.add_argument('--vocab_tree', default='colmap/vocab_tree.bin')
    parser.add_argument('--image_list', default='colmap/images/list.txt')
    parser.add_argument('--images', default='colmap/images')
    parser.add_argument('--sparse', default='colmap/colmap_sparse')
    parser.add_argument('--text', default='colmap/colmap_text')
    parser.add_argument('--static_sparse', default='colmap/static_sparse')
    parser.add_argument('--static_text', default='colmap/static_text')
    args = parser.parse_args()

    args.colmap_db = join(args.data_root, args.colmap_db)
    args.vocab_tree = join(args.data_root, args.vocab_tree)
    args.image_list = join(args.data_root, args.image_list)
    args.images = join(args.data_root, args.images)
    args.sparse = join(args.data_root, args.sparse)
    args.text = join(args.data_root, args.text)
    args.static_sparse = join(args.data_root, args.static_sparse)
    args.static_text = join(args.data_root, args.static_text)

    os.makedirs(join(args.static_sparse, '0'), exist_ok=True)
    os.makedirs(args.static_text, exist_ok=True)
    
    # maybe construct vocab tree
    if args.vocab_tree and not os.path.exists(args.vocab_tree):
        log(f'will construct a vocab tree at {args.vocab_tree}, this could take forever...', 'yellow')
        run(f'colmap vocab_tree_builder --database_path {args.colmap_db} --vocab_tree_path {args.vocab_tree}')
        
    # perform feature extraction
    run(f'colmap feature_extractor --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.camera_model {args.colmap_camera_model} --ImageReader.camera_params \"{args.colmap_camera_params}\" --database_path {args.colmap_db} --image_path {args.images} --image_list {args.image_list}')

    # try matching images
    match_cmd = f'colmap {args.colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {args.colmap_db} --VocabTreeMatching.match_list_path {args.image_list}'
    if args.vocab_tree:
        match_cmd += f' --VocabTreeMatching.vocab_tree_path {args.vocab_tree}'
    run(match_cmd)

    # perform registration
    run(f'colmap image_registrator --database_path {args.colmap_db} --input_path {args.sparse}/0 --output_path {args.static_sparse}/0')

    # global bundle adjustment for principal point
    tolerance = {
        '--BundleAdjustment.refine_principal_point': 1,
        '--BundleAdjustment.max_num_iterations': 20,
        '--BundleAdjustment.function_tolerance': 1e-6,
        '--BundleAdjustment.gradient_tolerance': 1e-10,
        '--BundleAdjustment.parameter_tolerance': 1e-8,
    }
    tolerance_string = ' '.join([f'{k} {v}' for k, v in tolerance.items()])
    log('running colmap bundle_adjuster, this could take a smaller forever, grab a coffee plz')
    run(f'colmap bundle_adjuster --input_path {args.static_sparse}/0 --output_path {args.static_sparse}/0 {tolerance_string}')
    # run(f"colmap model_orientation_aligner --image_path {args.images} --input_path {args.static_sparse}/0 --output_path {args.static_sparse}/0")
    run(f"colmap model_converter --input_path {args.static_sparse}/0 --output_path {args.static_text} --output_type TXT")
    correct_colmap_scale(args.static_sparse, args.static_text)


if __name__ == '__main__':
    main()
