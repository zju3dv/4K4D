#version 330
#pragma vscode_glsllint_stage : frag

// Input color from vertex shader
in vec3 vert_color;                          // albedo
in float vert_alpha;                         // alpha
layout(location = 0) out vec4 write_color;   // 16 bit * 4
layout(location = 1) out float write_upper;  // 32 bit, not used
layout(location = 2) out float write_lower;  // 32 bit

void main() {
    // Controls the shape of the point
    vec2 resd = gl_PointCoord - vec2(0.5);
    float dist = dot(resd, resd);  // compute the distance to the center of the point
    if (dist >= 0.25)              // skip this fragment for round points
        discard;

    // Render the final color
    write_color = vec4(vert_color, vert_alpha);  // write the color buffer
}
