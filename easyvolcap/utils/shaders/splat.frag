#version 330
#pragma vscode_glsllint_stage : frag

uniform mat4x4 K;
uniform mat4x4 M;
uniform mat4x4 V;
uniform int H;  // viewport size
uniform int W;
uniform int pass_index = 0;
uniform bool point_smooth = true;    // do we render round points or use sprites?
uniform bool alpha_blending = true;  // do we discard the rendering results of previous layers?

// Input color from vertex shader
in vec3 vert_color;   // albedo
in float vert_alpha;  // alpha

// Mesh rendering compositing
uniform sampler2D read_color;  // the composite color and weight from previous frame
uniform sampler2D read_upper;  // the composite depth from previous pass
uniform sampler2D read_lower;  // the composite depth from previous pass

layout(location = 0) out vec4 write_color;   // 16 bit * 4
layout(location = 1) out float write_upper;  // 32 bit, UNUSED: remove too far away points
layout(location = 2) out float write_lower;  // 32 bit

void main() {    // frag: pix-point
    float dist;  // shared variable

    if (point_smooth || alpha_blending) {
        // Controls the shape of the point
        vec2 resd = gl_PointCoord - vec2(0.5);
        dist = dot(resd, resd);  // compute the distance to the center of the point
        if (dist >= 0.25)        // skip this fragment for round points
            discard;
    }  // else, always render, no discard

    // Unavoidable cheap computations
    ivec2 screen_pts = ivec2(gl_FragCoord.xy);  // prepare the screen rendering coordinates
    float curr_depth = gl_FragCoord.z / gl_FragCoord.w;

    if (alpha_blending) {
        /** TODO: Control the density of the point
         *  If the total number of rendering pass is bigger than or equal to 10, always use the original blending model
         *  If the total number of rendering pass is just 1, use the solid rendering model, where the alpha value is always 1
         */
        float alpha = vert_alpha * (1 - dist * 4);  // this is the pixel weight (radius: 0.5)
        vec3 curr_rgb = vert_color * alpha;
        float curr_occ = alpha;

        // The last channel of color_buffer (rgba) is the accumulated transparency
        if (pass_index > 0) {  // 1fps boost
            // Depth peeling test
            float prev_lower = texelFetch(read_lower, screen_pts, 0).r;  // loads previous depth value, discard closer points
            if (curr_depth <= prev_lower)                            // skip this fragment if depth is smaller than previous pass
                discard;

            vec4 prev_color = texelFetch(read_color, screen_pts, 0);              // perform texture fetching
            curr_rgb = vert_color * alpha * (1 - prev_color.a) + prev_color.rgb;  // the actual volume rendering
            curr_occ = 1 - (1 - alpha) * (1 - prev_color.a);                      // accumulated opacity
        }

        write_color = vec4(curr_rgb, curr_occ);  // write the color buffer
        write_lower = curr_depth;            // prepare the next depth buffer to start the rendering
    } else {
        // Perform depth peeling
        if (pass_index > 0) {  // 1fps boost
            // Depth peeling test
            float prev_lower = texelFetch(read_lower, screen_pts, 0).r;  // loads previous depth value, discard closer points
            if (curr_depth <= prev_lower)                            // skip this fragment if depth is smaller than previous pass
                discard;
        }

        // Render the final layer's color only
        write_color = vec4(vert_color, vert_alpha);  // write the color buffer
        write_lower = curr_depth;                // prepare the next depth buffer to start the rendering
    }
}
