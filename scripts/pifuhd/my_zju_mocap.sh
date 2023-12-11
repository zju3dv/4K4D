data=data/my_zjumocap
dirs=(my_313 my_315 my_377 my_386 my_387 my_390 my_392 my_393 my_394)
for dir in "${dirs[@]}"; do
    python scripts/pifuhd/create_image_list_from_dir.py --input_dir $data/$dir --image_output $data/$dir/image_list.txt --mask_output $data/$dir/mask_list.txt
    python scripts/pifuhd/inference_pifuhd_normal.py --input_file $data/$dir/image_list.txt --input_mask $data/$dir/mask_list.txt --output_dir $data/$dir/normal --prefix_dir $data/$dir/images --chunk_size 1024
done
