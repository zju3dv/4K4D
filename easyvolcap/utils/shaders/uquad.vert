#version 330
#pragma vscode_glsllint_stage : vert

uniform sampler2D texture;
layout(location = 0) in vec3 position;

void main() {
    // Only passing through
    gl_Position = vec4(position, 1.0);  // doing a perspective projection to clip space
}