# Rotate light, static human, static view
# Ours
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 4, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" exp_name ablation_relight_synthetic_jody albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/synthetic_human/base_synthetic_malcolm.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 0, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" exp_name ablation_relight_synthetic_malcolm albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/mobile_stage/base_mobile_purple.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 6, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" exp_name ablation_relight_mobile_purple albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/mobile_stage/base_mobile_xuzhen.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 0, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" exp_name ablation_relight_mobile_xuzhen albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

# Brute force baseline
python run.py -t visualize -c configs/synthetic_human/brute_synthetic_jody.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 4, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/synthetic_human/brute_synthetic_malcolm.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 0, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/mobile_stage/brute_mobile_purple.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 6, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/mobile_stage/brute_mobile_xuzhen.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 0, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

# Relighting4D baseline
python run.py -t visualize -c configs/synthetic_human/neuralbody_synthetic_jody.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 4, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/synthetic_human/neuralbody_synthetic_malcolm.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 0, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/mobile_stage/neuralbody_mobile_purple.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 6, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/mobile_stage/neuralbody_mobile_xuzhen.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 0, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

# # NeRFactor baseline (only on the training frame)
# python run.py -t visualize -c configs/synthetic_human/nerf_synthetic_jody.yaml relighting True exp_name nerfactor_synthetic_jody_1f vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 15, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

# python run.py -t visualize -c configs/synthetic_human/nerf_synthetic_malcolm.yaml relighting True exp_name nerfactor_synthetic_malcolm_1f vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 19, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

# python run.py -t visualize -c configs/mobile_stage/nerf_mobile_purple.yaml relighting True exp_name nerfactor_mobile_purple_1f vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 8, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

# python run.py -t visualize -c configs/mobile_stage/nerf_mobile_xuzhen.yaml relighting True exp_name nerfactor_mobile_xuzhen_1f vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 14, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

# Static light, static human, rotate view
# Ours
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" exp_name ablation_relight_synthetic_jody albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/synthetic_human/base_synthetic_malcolm.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" exp_name ablation_relight_synthetic_malcolm albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/mobile_stage/base_mobile_purple.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" exp_name ablation_relight_mobile_purple albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/mobile_stage/base_mobile_xuzhen.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" exp_name ablation_relight_mobile_xuzhen albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

# Brute force baseline
python run.py -t visualize -c configs/synthetic_human/brute_synthetic_jody.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/synthetic_human/brute_synthetic_malcolm.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/mobile_stage/brute_mobile_purple.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/mobile_stage/brute_mobile_xuzhen.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

# Relighting4D baseline
python run.py -t visualize -c configs/synthetic_human/neuralbody_synthetic_jody.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/synthetic_human/neuralbody_synthetic_malcolm.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/mobile_stage/neuralbody_mobile_purple.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

python run.py -t visualize -c configs/mobile_stage/neuralbody_mobile_xuzhen.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

# # NeRFactor baseline (only on the training frame)
# python run.py -t visualize -c configs/synthetic_human/nerf_synthetic_jody.yaml relighting True exp_name nerfactor_synthetic_jody_1f vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

# python run.py -t visualize -c configs/synthetic_human/nerf_synthetic_malcolm.yaml relighting True exp_name nerfactor_synthetic_malcolm_1f vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

# python run.py -t visualize -c configs/mobile_stage/nerf_mobile_purple.yaml relighting True exp_name nerfactor_mobile_purple_1f vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

# python run.py -t visualize -c configs/mobile_stage/nerf_mobile_xuzhen.yaml relighting True exp_name nerfactor_mobile_xuzhen_1f vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

# Static light, dynamic human, static view
# Ours
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 4, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" exp_name ablation_relight_synthetic_jody albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

python run.py -t visualize -c configs/synthetic_human/base_synthetic_malcolm.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 0, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" exp_name ablation_relight_synthetic_malcolm albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

python run.py -t visualize -c configs/mobile_stage/base_mobile_purple.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 6, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" exp_name ablation_relight_mobile_purple albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

python run.py -t visualize -c configs/mobile_stage/base_mobile_xuzhen.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 0, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" exp_name ablation_relight_mobile_xuzhen albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

# Brute force baseline
python run.py -t visualize -c configs/synthetic_human/brute_synthetic_jody.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 4, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

python run.py -t visualize -c configs/synthetic_human/brute_synthetic_malcolm.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 0, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

python run.py -t visualize -c configs/mobile_stage/brute_mobile_purple.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 6, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

python run.py -t visualize -c configs/mobile_stage/brute_mobile_xuzhen.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 0, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

# Relighting4D baseline
python run.py -t visualize -c configs/synthetic_human/neuralbody_synthetic_jody.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 4, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

python run.py -t visualize -c configs/synthetic_human/neuralbody_synthetic_malcolm.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 0, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

python run.py -t visualize -c configs/mobile_stage/neuralbody_mobile_purple.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 6, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

python run.py -t visualize -c configs/mobile_stage/neuralbody_mobile_xuzhen.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 0, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

# # NeRFactor baseline (only on the training frame)
# python run.py -t visualize -c configs/synthetic_human/nerf_synthetic_jody.yaml relighting True exp_name nerfactor_synthetic_jody_1f vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 15, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose"

# python run.py -t visualize -c configs/synthetic_human/nerf_synthetic_malcolm.yaml relighting True exp_name nerfactor_synthetic_malcolm_1f vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 19, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose"

# python run.py -t visualize -c configs/mobile_stage/nerf_mobile_purple.yaml relighting True exp_name nerfactor_mobile_purple_1f vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 8, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose"

# python run.py -t visualize -c configs/mobile_stage/nerf_mobile_xuzhen.yaml relighting True exp_name nerfactor_mobile_xuzhen_1f vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 14, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose"

# To compare with NeRFactor, we need to render the training frame rotate light and training frame rotate view
# Ours
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 15, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" exp_name ablation_relight_synthetic_jody albedo_multiplier 2.0

python run.py -t visualize -c configs/synthetic_human/base_synthetic_malcolm.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 19, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" exp_name ablation_relight_synthetic_malcolm albedo_multiplier 2.0

python run.py -t visualize -c configs/mobile_stage/base_mobile_purple.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 8, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" exp_name ablation_relight_mobile_purple albedo_multiplier 2.0

python run.py -t visualize -c configs/mobile_stage/base_mobile_xuzhen.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 14, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" exp_name ablation_relight_mobile_xuzhen albedo_multiplier 2.0

# Brute force baseline
python run.py -t visualize -c configs/synthetic_human/brute_synthetic_jody.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 15, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

python run.py -t visualize -c configs/synthetic_human/brute_synthetic_malcolm.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 19, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

python run.py -t visualize -c configs/mobile_stage/brute_mobile_purple.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 8, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

python run.py -t visualize -c configs/mobile_stage/brute_mobile_xuzhen.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 14, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

# Relighting4D baseline
python run.py -t visualize -c configs/synthetic_human/neuralbody_synthetic_jody.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 15, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

python run.py -t visualize -c configs/synthetic_human/neuralbody_synthetic_malcolm.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 19, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

python run.py -t visualize -c configs/mobile_stage/neuralbody_mobile_purple.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 8, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

python run.py -t visualize -c configs/mobile_stage/neuralbody_mobile_xuzhen.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 14, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

# NeRFactor baseline (only on the training frame)
python run.py -t visualize -c configs/synthetic_human/nerf_synthetic_jody.yaml relighting True exp_name nerfactor_synthetic_jody_1f vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 15, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

python run.py -t visualize -c configs/synthetic_human/nerf_synthetic_malcolm.yaml relighting True exp_name nerfactor_synthetic_malcolm_1f vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 19, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

python run.py -t visualize -c configs/mobile_stage/nerf_mobile_purple.yaml relighting True exp_name nerfactor_mobile_purple_1f vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 8, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

python run.py -t visualize -c configs/mobile_stage/nerf_mobile_xuzhen.yaml relighting True exp_name nerfactor_mobile_xuzhen_1f vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 14, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static"

# Novel view for comparing with NeRFactor
# Ours
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" exp_name ablation_relight_synthetic_jody albedo_multiplier 2.0

python run.py -t visualize -c configs/synthetic_human/base_synthetic_malcolm.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" exp_name ablation_relight_synthetic_malcolm albedo_multiplier 2.0

python run.py -t visualize -c configs/mobile_stage/base_mobile_purple.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" exp_name ablation_relight_mobile_purple albedo_multiplier 2.0

python run.py -t visualize -c configs/mobile_stage/base_mobile_xuzhen.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view" exp_name ablation_relight_mobile_xuzhen albedo_multiplier 2.0

# Brute force baseline
python run.py -t visualize -c configs/synthetic_human/brute_synthetic_jody.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

python run.py -t visualize -c configs/synthetic_human/brute_synthetic_malcolm.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

python run.py -t visualize -c configs/mobile_stage/brute_mobile_purple.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

python run.py -t visualize -c configs/mobile_stage/brute_mobile_xuzhen.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

# Relighting4D baseline
python run.py -t visualize -c configs/synthetic_human/neuralbody_synthetic_jody.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

python run.py -t visualize -c configs/synthetic_human/neuralbody_synthetic_malcolm.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

python run.py -t visualize -c configs/mobile_stage/neuralbody_mobile_purple.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

python run.py -t visualize -c configs/mobile_stage/neuralbody_mobile_xuzhen.yaml relighting True vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

# NeRFactor baseline (only on the training frame)
python run.py -t visualize -c configs/synthetic_human/nerf_synthetic_jody.yaml relighting True exp_name nerfactor_synthetic_jody_1f vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

python run.py -t visualize -c configs/synthetic_human/nerf_synthetic_malcolm.yaml relighting True exp_name nerfactor_synthetic_malcolm_1f vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

python run.py -t visualize -c configs/mobile_stage/nerf_mobile_purple.yaml relighting True exp_name nerfactor_mobile_purple_1f vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

python run.py -t visualize -c configs/mobile_stage/nerf_mobile_xuzhen.yaml relighting True exp_name nerfactor_mobile_xuzhen_1f vis_novel_light True vis_novel_view True num_eval_frame 1 num_render_view 100 vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "view"

# Render ours without background sampling for main lighting condition
# Static light, dynamic human, static view
# Ours
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 4, vis_ground_shading True test_light '["main"]' ratio 0.33 store_video_output True extra_prefix "pose" exp_name ablation_relight_synthetic_jody albedo_multiplier 1.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300 ground_attach_envmap False

python run.py -t visualize -c configs/synthetic_human/base_synthetic_malcolm.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 0, vis_ground_shading True test_light '["main"]' ratio 0.33 store_video_output True extra_prefix "pose" exp_name ablation_relight_synthetic_malcolm albedo_multiplier 1.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300 ground_attach_envmap False

python run.py -t visualize -c configs/mobile_stage/base_mobile_purple.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 6, vis_ground_shading True test_light '["main"]' ratio 0.33 store_video_output True extra_prefix "pose" exp_name ablation_relight_mobile_purple albedo_multiplier 1.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300 ground_attach_envmap False

python run.py -t visualize -c configs/mobile_stage/base_mobile_xuzhen.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 0, vis_ground_shading True test_light '["main"]' ratio 0.33 store_video_output True extra_prefix "pose" exp_name ablation_relight_mobile_xuzhen albedo_multiplier 1.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300 ground_attach_envmap False

# Ours
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 4, vis_ground_shading True test_light '["gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" exp_name ablation_relight_synthetic_jody albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

python run.py -t visualize -c configs/synthetic_human/base_synthetic_malcolm.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 0, vis_ground_shading True test_light '["gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" exp_name ablation_relight_synthetic_malcolm albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

python run.py -t visualize -c configs/mobile_stage/base_mobile_purple.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 6, vis_ground_shading True test_light '["gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" exp_name ablation_relight_mobile_purple albedo_multiplier 3.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300

python run.py -t visualize -c configs/mobile_stage/base_mobile_xuzhen.yaml relighting True vis_novel_light True vis_pose_sequence True num_eval_frame 100 frame_interval 3 test_view 0, vis_ground_shading True test_light '["gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "pose" exp_name ablation_relight_mobile_xuzhen albedo_multiplier 3.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 300


# Render videos on another character for better visualization
# Matching Static novel pose
# Ours
python run.py -t visualize -c configs/synthetic_human/base_synthetic_josh.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 4, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" albedo_multiplier 2.0 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

# Brute force baseline
python run.py -t visualize -c configs/synthetic_human/brute_synthetic_josh.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 4, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440

# Relighting4D baseline
python run.py -t visualize -c configs/synthetic_human/neuralbody_synthetic_josh.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 4, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.33 store_video_output True extra_prefix "static" test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz begin_ith_frame 440


# Matching NeRFactor
# Ours
python run.py -t visualize -c configs/synthetic_human/base_synthetic_josh.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 15, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.5 store_video_output True extra_prefix "static" albedo_multiplier 2.0

# Brute force baseline
python run.py -t visualize -c configs/synthetic_human/brute_synthetic_josh.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 15, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.5 store_video_output True extra_prefix "static"


# Relighting4D baseline
python run.py -t visualize -c configs/synthetic_human/neuralbody_synthetic_josh.yaml relighting True vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 15, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.5 store_video_output True extra_prefix "static"

# NeRFactor baseline
python run.py -t visualize -c configs/synthetic_human/nerf_synthetic_josh.yaml relighting True exp_name nerfactor_synthetic_josh_1f vis_novel_light True vis_rotate_light True vis_pose_sequence True num_eval_frame 1 test_view 15, vis_ground_shading True test_light '["main", "gym_entrance", "shanghai_bund", "peppermint_powerplant_blue", "pink_sunrise", "olat0002-0027", "olat0004-0019"]' ratio 0.5 store_video_output True extra_prefix "static"