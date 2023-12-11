#version 330
#pragma vscode_glsllint_stage : frag

uniform sampler2D tex;
layout(location = 0) out vec4 frag_color;

void main() {
    // Only passing through
    frag_color = texelFetch(tex, ivec2(gl_FragCoord.xy), 0);
}