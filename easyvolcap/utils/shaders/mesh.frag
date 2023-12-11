#version 330
#pragma vscode_glsllint_stage : frag

uniform bool shade_flat;
uniform bool render_normal;

in vec3 world_frag_pos;  // Note: this is in world space
in vec3 cam_frag_pos;    // Note: this is in camera space
in vec3 vert_color;      // albedo
in vec3 vert_normal;     // normal

layout(location = 0) out vec4 frag_color;
// layout(location = 1) out vec4 frag_depth;

void main() {
    // Prepare normals
    if (render_normal) {
        vec3 shade_normal = vec3(0, 0, 1);
        if (shade_flat) {
            vec3 x_tangent = dFdx(vec3(cam_frag_pos));
            vec3 y_tangent = dFdy(vec3(cam_frag_pos));
            shade_normal = normalize(cross(x_tangent, y_tangent));  // already vec3
        } else {
            shade_normal = vert_normal;  // smoothed vertex normals
        }

        frag_color = vec4(shade_normal * 0.5 + vec3(0.5), 1.0);  // transform [-1,1] to [0, 1]
    } else {
        frag_color = vec4(vert_color, 1.0);
    }
    // vec4 vPack = vec4(1.0f, 256.0f, 65536.0, 16777216.0f);
    // frag_depth = vPack * length(cam_frag_pos);  // defined in linear space
    // gl_FragDepth = length(cam_frag_pos.xyz);  // for depth testing in linear space and easiler output
}