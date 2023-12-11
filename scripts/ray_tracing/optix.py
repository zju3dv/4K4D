# import optix
# from numba import *
# # from numba import cuda, float32, make_float2, fast_powf, make_uchar4, quantizeUnsigned8Bits, uint8


# @cuda.jit(device=True, fast_math=True)
# def computeRay(idx, dim):
#     U = params.cam_u
#     V = params.cam_v
#     W = params.cam_w
#     # Normalizing coordinates to [-1.0, 1.0]
#     d = float32(2.0) * make_float2(
#         float32(idx.x) / float32(dim.x), float32(idx.y) / float32(dim.y)
#     ) - float32(1.0)

#     origin = params.cam_eye
#     direction = normalize(d.x * U + d.y * V + W)
#     return origin, direction


# @cuda.jit(device=True, fast_math=True)
# def setPayload(p):
#     optix.SetPayload_0(float_as_int(p.x))
#     optix.SetPayload_1(float_as_int(p.y))
#     optix.SetPayload_2(float_as_int(p.z))


# def __closesthit__ch():
#     # When a built-in triangle intersection is used, a number of fundamental
#     # attributes are provided by the NVIDIA OptiX API, including barycentric coordinates.
#     barycentrics = optix.GetTriangleBarycentrics()

#     setPayload(make_float3(barycentrics, float32(1.0)))


# def __miss__ms():
#     miss_data = MissDataStruct(optix.GetSbtDataPointer())
#     setPayload(miss_data.bg_color)


# @cuda.jit(device=True, fast_math=True)
# def toSRGB(c):
#     # Use float32 for constants
#     invGamma = float32(1.0) / float32(2.4)
#     powed = make_float3(
#         fast_powf(c.x, invGamma),
#         fast_powf(c.y, invGamma),
#         fast_powf(c.z, invGamma),
#     )
#     return make_float3(
#         float32(12.92) * c.x
#         if c.x < float32(0.0031308)
#         else float32(1.055) * powed.x - float32(0.055),
#         float32(12.92) * c.y
#         if c.y < float32(0.0031308)
#         else float32(1.055) * powed.y - float32(0.055),
#         float32(12.92) * c.z
#         if c.z < float32(0.0031308)
#         else float32(1.055) * powed.z - float32(0.055),
#     )


# @cuda.jit(device=True, fast_math=True)
# def make_color(c):
#     srgb = toSRGB(clamp(c, float32(0.0), float32(1.0)))

#     return make_uchar4(
#         quantizeUnsigned8Bits(srgb.x),
#         quantizeUnsigned8Bits(srgb.y),
#         quantizeUnsigned8Bits(srgb.z),
#         uint8(255),
#     )


# def __raygen__rg():
#     # Look up your location within the launch grid
#     idx = optix.GetLaunchIndex()
#     dim = optix.GetLaunchDimensions()

#     # Map your launch idx to a screen location and create a ray from the camera
#     # location through the screen
#     ray_origin, ray_direction = computeRay(make_uint3(idx.x, idx.y, 0), dim)

#     # In __raygen__rg
#     payload_pack = optix.Trace(
#         params.handle,
#         ray_origin,
#         ray_direction,
#         float32(0.0),  # Min intersection distance
#         float32(1e16),  # Max intersection distance
#         float32(0.0),  # rayTime -- used for motion blur
#         OptixVisibilityMask(255),  # Specify always visible
#         uint32(OPTIX_RAY_FLAG_NONE),
#         uint32(0),  # SBT offset   -- Refer to OptiX Manual for SBT
#         uint32(1),  # SBT stride   -- Refer to OptiX Manual for SBT
#         uint32(0),  # missSBTIndex -- Refer to OptiX Manual for SBT
#     )

#     # In __raygen__rg
#     result = make_float3(
#         int_as_float(payload_pack.p0),
#         int_as_float(payload_pack.p1),
#         int_as_float(payload_pack.p2),
#     )
#     # Record results in your output raster
#     params.image[idx.y * params.image_width + idx.x] = make_color(result)
