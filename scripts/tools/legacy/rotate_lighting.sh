cmd='python run.py -t visualize -c configs/$dataset/$human.yaml relighting True vis_novel_light True vis_rotate_light True vis_ground_shading True test_view $test_view ratio $ratio test_light "${light}" ground_attach_envmap True'
light='["main", "gym_entrance", "peppermint_powerplant_blue", "shanghai_bund", "olat0002-0027"]'

dataset=mobile_stage
human=base_mobile_xuzhen
ratio=0.33
test_view='0,'

human=base_mobile_xuzhen
echo $(eval "echo $cmd")
eval $cmd
human=base_mobile_coat
eval $cmd
human=base_mobile_purple
eval $cmd
human=base_mobile_black
eval $cmd
human=base_mobile_white
eval $cmd
human=base_mobile_dress
eval $cmd
human=base_mobile_move
eval $cmd

dataset=synthetic_human
human=base_synthetic_jody
ratio=0.5
test_view='5,'

human=base_synthetic_jody
eval $cmd
human=base_synthetic_josh
eval $cmd
human=base_synthetic_leonard
eval $cmd
human=base_synthetic_manuel
eval $cmd
human=base_synthetic_malcolm
eval $cmd
human=base_synthetic_megan
eval $cmd
human=base_synthetic_nathan
eval $cmd
