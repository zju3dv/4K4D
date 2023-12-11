import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/genebody_test10')
    parser.add_argument('--scene', type=str, default='fuzhizhi')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--n_bones', type=int, default=24)
    args = parser.parse_args()

    # Fetch arguments from parser
    data_root = args.data_root
    scene = args.scene
    n_bones = args.n_bones

    # Define input and output directories
    input_dir = os.path.join(data_root, scene)
    output_dir = args.output_dir

    # Run idr pipeline of easymocap
    os.system(f'source scripts/zjumocap/zjumocap_easymocap.sh')
    os.system(f'easymocap {input_dir} {output_dir}')

    # Prepare motion
    os.system(f'python3 scripts/tools/prepare_motion.py --data_root {data_root} --human {scene} --n_bones {n_bones}')


if __name__ == "__main__":
    main()
