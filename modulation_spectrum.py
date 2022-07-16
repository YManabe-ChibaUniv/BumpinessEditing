from PIL import Image
from PIL import ImageDraw
import numpy as np
import os
import math
import params as prm

def max_contrast_sensitivity_ratio() -> float:
    max_ratio = 1.0
    for f in range(0, 255):
        w = (prm.A_K + prm.A_a * (f ** prm.A_c)) * (math.e ** (-1 * prm.A_b * f)) / prm.A_K
        if max_ratio < w:
            max_ratio = w
    return max_ratio

def modulation_gray(filename):
    img = Image.open("./img/" + filename).convert("L")
    f_xy = np.asarray(img)
    f_uv = np.fft.fft2(f_xy)
    shifted_f_uv = np.fft.fftshift(f_uv)
    x_pass_filter = Image.new(mode="L", size=(shifted_f_uv.shape[0], shifted_f_uv.shape[1]), color=0)
    draw = ImageDraw.Draw(x_pass_filter)
    center = (shifted_f_uv.shape[0] // 2, shifted_f_uv.shape[1] // 2)
    ellipse_pos = (center[0] - prm.outside_r, center[1] - prm.outside_r, center[0] + prm.outside_r, center[1] + prm.outside_r)
    draw.ellipse(ellipse_pos, fill=255)
    ellipse_pos = (center[0] - prm.inside_r, center[1] - prm.inside_r, center[0] + prm.inside_r, center[1] + prm.inside_r)
    draw.ellipse(ellipse_pos, fill=0)
    filter_array = np.asarray(x_pass_filter)

    # 保存先作成
    out_dir = "./result/" + os.path.splitext(filename)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += "/"
    mkdir_name = out_dir + "method1"
    if not os.path.exists(mkdir_name):
        os.makedirs(mkdir_name)
    mkdir_name = out_dir + "method2"
    if not os.path.exists(mkdir_name):
        os.makedirs(mkdir_name)

    gray_signal = []
    shifted_f_uv_ms = shifted_f_uv.copy()
    for f in prm.scalefactor:
        for j in range(filter_array.shape[0]):
            for i in range(filter_array.shape[1]):
                if (filter_array[j][i] == 255):
                    shifted_f_uv_ms[j][i] = shifted_f_uv[j][i] * f
                else:
                    shifted_f_uv_ms[j][i] = shifted_f_uv[j][i]
        unshifted_f_uv = np.fft.fftshift(shifted_f_uv_ms)
        i_f_xy = np.fft.ifft2(unshifted_f_uv).real
        gray_signal.append((Image.fromarray(i_f_xy).convert("L"), "{:.2f}".format(f), 1))

    # コントラスト感度
    filter_array_cont = np.zeros(shape=(shifted_f_uv.shape[0], shifted_f_uv.shape[1]), dtype=np.float32)
    center = (filter_array_cont.shape[0] // 2, filter_array_cont.shape[1] // 2)
    angle = 3.416
    max_f_ratio = max_contrast_sensitivity_ratio()
    for rate in prm.scalefactor:
        for j in range(filter_array_cont.shape[1]):
            for i in range(filter_array_cont.shape[0]):
                f = math.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2) / angle
                if rate > 1.0:
                    w = (prm.A_K + prm.A_a * (f ** prm.A_c)) * (math.e ** (-1 * prm.A_b * f)) / prm.A_K
                    filter_array_cont[j][i] = w
                    reg_ratio = (w - 1.0 / max_f_ratio - 1.0) * (rate - 1.0) + 1.0
                else:
                    _rate = 1.0 + (1.0 - rate)
                    w = (prm.A_K + prm.A_a * (f ** prm.A_c)) * (math.e ** (-1 * prm.A_b * f)) / prm.A_K
                    filter_array_cont[j][i] = w
                    reg_ratio = 1.0 / ((w - 1.0 / max_f_ratio - 1.0) * (_rate - 1.0) + 1.0)
                if filter_array_cont[j][i] >= 1.0 and filter_array[j][i] == 255:
                    shifted_f_uv_ms[j][i] = shifted_f_uv[j][i] * reg_ratio
                else:
                    shifted_f_uv_ms[j][i] = shifted_f_uv[j][i]
        unshifted_f_uv = np.fft.fftshift(shifted_f_uv_ms)
        i_f_xy = np.fft.ifft2(unshifted_f_uv).real
        # 平均輝度の差分を加算する ----------------------
        src_mean = f_xy.mean()
        edited_mean = i_f_xy.mean()
        i_f_xy = i_f_xy + (src_mean - edited_mean)
        # -----------------------------------------------
        gray_signal.append((Image.fromarray(i_f_xy).convert("L"), "{:.2f}".format(rate), 2))

    return gray_signal

def modulation_color_using_gray(filename, gray) -> None:
    color_space = "YCbCr"
    _, cb, cr = Image.open("./img/" + filename).convert(color_space).split()

    out_dir = "./result/" + os.path.splitext(filename)[0] + "/"
    for (signal, string, method) in gray:
        out = Image.merge(color_space, (signal, cb, cr)).convert("RGB")
        if method == 1:
            out.save(out_dir + "method1/" + string + ".png")
        else:
            out.save(out_dir + "method2/" + string + ".png")

if __name__ == '__main__':
    files = os.listdir("./img")
    files = ["tex001.bmp", "tex002.bmp"]
    for file in files:
        print(file)
        gray = modulation_gray(file)
        modulation_color_using_gray(file, gray)