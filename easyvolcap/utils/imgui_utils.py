import numpy as np
from glm import vec3, vec4, mat3, mat4, mat4x3
from imgui_bundle import imgui, imguizmo, imgui_toggle, immvision, portable_file_dialogs, implot, ImVec2, ImVec4


def push_button_color(col: np.uint32 = 0xff5533ff):  # 0xff3355ff
    a = col >> 24 & 0xff
    b = col >> 16 & 0xff
    g = col >> 8 & 0xff
    r = col >> 0 & 0xff

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


def colored_wrapped_text(col: np.uint32 = 0xff5533ff, text: str = 'some message'):
    a = col >> 24 & 0xff
    b = col >> 16 & 0xff
    g = col >> 8 & 0xff
    r = col >> 0 & 0xff
    col = ImVec4(r / 255, g / 255, b / 255, a / 255)

    imgui.push_text_wrap_pos(0.0)
    imgui.push_style_color(imgui.Col_.text, col)
    imgui.text(text)
    imgui.pop_style_color()
    imgui.pop_text_wrap_pos()
