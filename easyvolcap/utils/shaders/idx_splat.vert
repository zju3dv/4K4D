#version 330
#pragma vscode_glsllint_stage : vert

uniform mat4x4 K;
uniform mat4x4 M;
uniform mat4x4 V;
uniform int H;  // viewport size
uniform int W;

layout(location = 0) in vec3 verts;
layout(location = 1) in float radius;
flat out int vert_index;  // pass through

void main() {
    vert_index = gl_VertexID;
    gl_Position = K * V * M * vec4(verts, 1.0);                // doing a perspective projection to clip space
    gl_PointSize = abs(H * K[1][1] * radius / gl_Position.w);  // need to determine size in pixels
    // https://stackoverflow.com/questions/25780145/gl-pointsize-corresponding-to-world-space-size
}