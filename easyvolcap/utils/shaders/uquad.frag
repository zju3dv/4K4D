#version 330
#pragma vscode_glsllint_stage : frag

uniform bool use_tex0 = false;
uniform bool use_tex1 = false;
uniform bool use_tex2 = false;

uniform vec4 value0 = vec4(0.0);
uniform vec4 value1 = vec4(0.0);
uniform vec4 value2 = vec4(0.0);

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;

layout(location = 0) out vec4 out0;
layout(location = 1) out vec4 out1;
layout(location = 2) out vec4 out2;

void main() {
    // Only passing through
    out0 = use_tex0 ? texelFetch(tex0, ivec2(gl_FragCoord.xy), 0) : value0;
    out1 = use_tex1 ? texelFetch(tex1, ivec2(gl_FragCoord.xy), 0) : value1;
    out2 = use_tex2 ? texelFetch(tex2, ivec2(gl_FragCoord.xy), 0) : value2;
}