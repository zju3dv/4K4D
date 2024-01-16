import torch
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.nerf_utils import volume_rendering


def pad_layer_output(layer_output: dotdict):
    # Fetch the layer output
    HrWr, xywh = layer_output.HrWr, layer_output.xywh
    # Complete the foreground layer object bounding box
    output = dotdict()

    for key, value in layer_output.items():
        # Complete a specific output of all layers
        if key in ['HrWr', 'xywh', 'layer_render']: continue

        # Nasty shape, especially in layered representation
        H, W = HrWr[-1]  # !: H, W should be the same for all foreground layers
        B, _, N, C = value[-1].shape if len(value[-1].shape) == 4 else value[-1][..., None].shape  # N may differ for different foreground layers
        x, y, w, h = xywh[-1]  # x, y, w, h of the foreground object bounding box, may differ

        # Actual complete logic
        value_c = torch.zeros(B, H, W, N, C).to(value[-1].device)  # (B, H, W, N, C)
        value_c[:, y:y + h, x:x + w] = value[-1].reshape(B, h, w, N, C)  # (B, H, W, N, C)
        output[key] = value_c.reshape(B, -1, N, C)

    return output


def pad_layers_output(layer_output: dotdict):
    # Fetch the layer output
    HrWr, xywh = layer_output.HrWr, layer_output.xywh
    # Complete the foreground layer object bounding box
    output = dotdict()

    # Concatenate all layers output
    for key, value in layer_output.items():
        # Complete a specific output of all layers
        if key in ['HrWr', 'xywh', 'layer_render']: continue
        else: output[key] = []

        for i in range(len(value)):
            # Nasty shape, especially in layered representation
            H, W = HrWr[i]  # !: H, W should be the same for all foreground layers
            B, _, N, C = value[i].shape if len(value[i].shape) == 4 else value[i][..., None].shape  # N may differ for different foreground layers
            x, y, w, h = xywh[i]  # x, y, w, h of the foreground object bounding box, may differ

            # Actual complete logic
            value_c = torch.zeros(B, H, W, N, C).to(value[i].device)  # (B, H, W, N, C)
            value_c[:, y:y + h, x:x + w] = value[i].reshape(B, h, w, N, C)  # (B, H, W, N, C)
            output[key].append(value_c.reshape(B, -1, N, C))

        # Concatenate all layers output together
        output[key] = torch.cat(output[key], dim=-2)  # (B, H * W, N * num_fg_layer, C)

    # Sort the z_vals and get the indices
    output.z_vals, indices = torch.sort(output.z_vals, dim=-2)  # (B, H * W, N * num_fg_layer, 1)
    # Rearrange the other outputs according to the indices
    for key, value in output.items():
        if key in ['z_vals', 'HrWr', 'xywh', 'layer_render']: continue
        else: output[key] = torch.gather(value, dim=-2, index=indices.expand(-1, -1, -1, value.shape[-1]))

    return output


def volume_rendering_layer(output: dotdict, rgb: torch.Tensor, occ: torch.Tensor, bg_brightness=0.0):
    # See volume_rendering for more intuition
    # Concatenate the background rgb and occ to the last
    rgbs = torch.cat([output.rgb, rgb], dim=-2)  # (B, H * W, N * num_fg_layer + N, 3)
    occs = torch.cat([output.occ, occ], dim=-2)  # (B, H * W, N * num_fg_layer + N, 3)

    weights, rgb_map, acc_map = volume_rendering(rgbs, occs, bg_brightness)  # (B, H * W, N * num_fg_layer + N), (B, H * W, 3), (B, H * W, 1)
    return weights, rgb_map, acc_map
