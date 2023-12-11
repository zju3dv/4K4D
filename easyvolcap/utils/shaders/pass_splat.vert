#version 330
#pragma vscode_glsllint_stage : vert

uniform mat4x4 K;
uniform mat4x4 M;
uniform mat4x4 V;
uniform int H;  // viewport size
uniform int W;

layout(location = 0) in vec4 pix_verts;    // ndc points
layout(location = 1) in vec3 pix_color;    // rgb
layout(location = 2) in float pix_radius;  // ndc point radius
out vec3 vert_color;                       // pass through

void main() {
    vert_color = pix_color;     // passing through
    gl_Position = pix_verts;    // doing a perspective projection to clip space
    gl_PointSize = pix_radius;  // need to determine size in pixels
    // https://stackoverflow.com/questions/25780145/gl-pointsize-corresponding-to-world-space-size
}