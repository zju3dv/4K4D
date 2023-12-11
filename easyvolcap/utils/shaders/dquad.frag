#version 330
#pragma vscode_glsllint_stage : frag

uniform sampler2D read_color;
uniform sampler2D read_upper;
uniform sampler2D read_lower;

void main() {
    // Only passing through
    gl_FragDepth = texelFetch(read_upper, ivec2(gl_FragCoord.xy), 0).r;
}