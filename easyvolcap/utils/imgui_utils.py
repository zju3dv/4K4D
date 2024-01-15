import numpy as np
from glm import vec3, vec4, mat3, mat4, mat4x3
from imgui_bundle import imgui, imguizmo, imgui_toggle, immvision, portable_file_dialogs, implot, ImVec2, ImVec4


def col2imu32(col: np.uint32 = 0xff3355ff):
    r = col >> 24 & 0xff
    g = col >> 16 & 0xff
    b = col >> 8 & 0xff
    a = col >> 0 & 0xff
    return np.uint32(a << 24 | b << 16 | g << 8 | r)


def col2rgba(col: np.uint32 = 0xff3355ff):
    r = col >> 24 & 0xff
    g = col >> 16 & 0xff
    b = col >> 8 & 0xff
    a = col >> 0 & 0xff
    return r, g, b, a


def col2vec4(col: np.uint32 = 0xff3355ff):
    r = col >> 24 & 0xff
    g = col >> 16 & 0xff
    b = col >> 8 & 0xff
    a = col >> 0 & 0xff
    return ImVec4(r / 255, g / 255, b / 255, a / 255)


def vec42col(col: ImVec4):
    r = int(col.x * 255)
    g = int(col.y * 255)
    b = int(col.z * 255)
    a = int(col.w * 255)
    return np.uint32(r << 24 | g << 16 | b << 8 | a)


def list2col(col: list):
    r = int(col[0] * 255)
    g = int(col[1] * 255)
    b = int(col[2] * 255)
    a = int(col[3] * 255)
    return np.uint32(r << 24 | g << 16 | b << 8 | a)


def push_button_color(col: np.uint32 = 0xff3355ff):
    r, g, b, a = col2rgba(col)

    main = ImVec4(r / 255, g / 255, b / 255, a / 255)
    hovered = ImVec4(min(r / 255 + 0.1, 1), min(g / 255 + 0.1, 1), min(b / 255 + 0.1, 1), a / 255)
    active = ImVec4(min(r / 255 + 0.05, 1), min(g / 255 + 0.05, 1), min(b / 255 + 0.05, 1), a / 255)

    # imgui.push_id(0)
    imgui.push_style_color(imgui.Col_.button, main)
    imgui.push_style_color(imgui.Col_.button_hovered, hovered)
    imgui.push_style_color(imgui.Col_.button_active, active)


def pop_button_color():
    imgui.pop_style_color()
    imgui.pop_style_color()
    imgui.pop_style_color()
    # imgui.pop_id()


def tooltip(content: str):
    imgui.same_line()
    imgui.text_disabled('(?)')
    if imgui.is_item_hovered():
        imgui.set_tooltip(content)


def colored_wrapped_text(col: np.uint32 = 0xff3355ff, text: str = 'some message'):
    col = col2vec4(col)

    imgui.push_text_wrap_pos(0.0)
    imgui.push_style_color(imgui.Col_.text, col)
    imgui.text(text)
    imgui.pop_style_color()
    imgui.pop_text_wrap_pos()
