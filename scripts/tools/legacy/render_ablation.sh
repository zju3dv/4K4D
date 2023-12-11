# Visibility

# Ours
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_normal_map True vis_rendering_map True vis_shading_map True vis_albedo_map True albedo_multiplier 2.0 begin_ith_frame 42 num_eval_frame 1 test_view 0, replace_light olat0004-0019 tonemapping_albedo True exp_name ablation_relight_synthetic_jody vis_ground_shading True

python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_normal_map True vis_rendering_map True vis_shading_map True vis_albedo_map True albedo_multiplier 2.0 begin_ith_frame 42 num_eval_frame 1 test_view 0, replace_light olat0004-0017 tonemapping_albedo True exp_name ablation_relight_synthetic_jody vis_ground_shading True


# w/o soft
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_normal_map True vis_rendering_map True vis_shading_map True vis_albedo_map True albedo_multiplier 2.0 begin_ith_frame 42 num_eval_frame 1 test_view 0, replace_light olat0004-0019 tonemapping_albedo True exp_name ablation_relight_synthetic_jody vis_ground_shading True no_dfss True

python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_normal_map True vis_rendering_map True vis_shading_map True vis_albedo_map True albedo_multiplier 2.0 begin_ith_frame 42 num_eval_frame 1 test_view 0, replace_light olat0004-0017 tonemapping_albedo True exp_name ablation_relight_synthetic_jody vis_ground_shading True no_dfss True


# w/o visibility
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_normal_map True vis_rendering_map True vis_shading_map True vis_albedo_map True albedo_multiplier 2.0 begin_ith_frame 42 num_eval_frame 1 test_view 0, replace_light olat0004-0019 tonemapping_albedo True exp_name ablation_relight_synthetic_jody vis_ground_shading True local_visibility True

python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_normal_map True vis_rendering_map True vis_shading_map True vis_albedo_map True albedo_multiplier 2.0 begin_ith_frame 42 num_eval_frame 1 test_view 0, replace_light olat0004-0017 tonemapping_albedo True exp_name ablation_relight_synthetic_jody vis_ground_shading True local_visibility True


# HDQ
# World-space Hierarchical Distance Query Sphere Tracing
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_normal_map True vis_rendering_map True vis_shading_map True vis_albedo_map True albedo_multiplier 2.0 begin_ith_frame 42 num_eval_frame 1 test_view 0, replace_light gym_entrance tonemapping_albedo True exp_name ablation_relight_synthetic_jody

# World-space-canonical-distance Sphere Tracing
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_normal_map True vis_rendering_map True vis_shading_map True vis_albedo_map True albedo_multiplier 2.0 begin_ith_frame 42 num_eval_frame 1 test_view 0, replace_light gym_entrance tonemapping_albedo True exp_name ablation_relight_synthetic_jody ablate_hdq_mode world obj_lvis.iter 8

# Visibility for material reconstruction -> baked shadow on albedo
python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_normal_map True vis_rendering_map True vis_shading_map True vis_albedo_map True albedo_multiplier 2.0 begin_ith_frame 0 num_eval_frame 1 test_view 4, tonemapping_albedo True exp_name ablation_relight_synthetic_jody # use the albedo output

python run.py -t visualize -c configs/synthetic_human/base_synthetic_jody.yaml relighting True vis_normal_map True vis_rendering_map True vis_shading_map True vis_albedo_map True albedo_multiplier 2.0 begin_ith_frame 0 num_eval_frame 1 test_view 4, tonemapping_albedo True exp_name ablation_relight_synthetic_jody_local_visibility local_visibility True # use the albedo output