#version 330
#pragma vscode_glsllint_stage : frag

uniform mat4x4 K;
uniform mat4x4 M;
uniform mat4x4 V;
uniform int H;  // viewport size
uniform int W;

// Input color from vertex shader
flat in int vert_index;  // pass through

// Mesh rendering compositing
uniform isampler2D read_index;  // the index of the previous depth peeling pass
uniform sampler2D read_lower;   // the composite weight from previous pass

layout(location = 0) out int write_index;    // 32 bit
layout(location = 1) out float write_lower;  // 32 bit

void main() {
    // Unavoidable cheap computations
    ivec2 screen_pts = ivec2(gl_FragCoord.xy);  // prepare the screen rendering coordinates

    // Controls the shape of the point
    vec2 resd = gl_PointCoord - vec2(0.5);
    float dist = dot(resd, resd);  // compute the distance to the center of the point
    if (dist >= 0.25)              // skip this fragment for round points
        discard;

    // // Depth peeling test
    // int prev_index = texelFetch(read_index, screen_pts, 0).r;  // loads previous depth value, discard closer points
    // if (prev_index == vert_index)                              // skip this fragment if depth is smaller than previous pass
    //     discard;

    // Depth peeling test
    float prev_lower = texelFetch(read_lower, screen_pts, 0).r;  // loads previous depth value, discard closer points
    if (gl_FragCoord.z <= prev_lower)                            // skip this fragment if depth is smaller than previous pass
        discard;

    // The last channel of color_buffer (rgba) is the accumulated transparency
    write_index = vert_index;      // store the vertex index of this depth pass
    write_lower = gl_FragCoord.z;  // prepare the next depth buffer to start the rendering
}