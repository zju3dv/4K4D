#version 330
#pragma vscode_glsllint_stage : vert

uniform mat4x4 K;
uniform mat4x4 M;
uniform mat4x4 V;
uniform mat4x4 P;  // P = K * V * M, this is the projection matrix
uniform int H;     // viewport size
uniform int W;
uniform float n;
uniform float f;
uniform int pass_index = 0;
uniform float radii_mult = 1.0;       // Controlling the radius expansion
uniform float alpha_thresh = 0.0001;  // Discard points with low density early on

layout(location = 0) in vec3 verts;
layout(location = 1) in vec3 color;  // rgb
layout(location = 2) in float radius;
layout(location = 3) in float alpha;
out vec3 vert_color;   // pass through
out float vert_alpha;  // pass through

void main() {
    vert_color = color;                                                     // passing through
    vert_alpha = alpha;                                                     // passing through
    gl_Position = P * vec4(verts, 1.0);                                     // doing a perspective projection to clip space
    gl_PointSize = abs(H * K[1][1] * radius / gl_Position.w) * radii_mult;  // need to determine size in pixels
    // https://stackoverflow.com/questions/25780145/gl-pointsize-corresponding-to-world-space-size

    // Discard transparent points
    if (alpha <= alpha_thresh)                // this gives two more fps
        gl_Position.z = 100 * gl_Position.w;  // effectively discard this
}