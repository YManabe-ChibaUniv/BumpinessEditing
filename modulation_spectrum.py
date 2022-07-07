from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
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

def save_image(f_xy, filter_array, magnitude, l):
    fig, axes = plt.subplots(1, 3)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
        # filtered image
        axes[0].imshow(f_xy, cmap='gray')
        axes[0].set_title("filtered image " + str(l))
        # filter
        axes[1].imshow(filter_array, cmap='gray')
        axes[1].set_title("filter " + str(l))
        # magnitude
        axes[2].imshow(magnitude, cmap='gray')
        axes[2].set_title("magnitude " + str(l))
        fig.tight_layout()
        fig.savefig("./result/multiband/filtered" + str(l) + ".png")
        plt.close()

def multiscale(filename):
    # load image
    img = Image.open(filename).convert("L")
    f_xy = np.asarray(img)

    # フーリエ変換
    f_uv = np.fft.fft2(f_xy)
    shifted_f_uv = np.fft.fftshift(f_uv)

    # make filter
    # multiscale image num
    n = 6
    x_pass_filter = Image.new(mode="L", size=(shifted_f_uv.shape[0], shifted_f_uv.shape[1]), color=0)
    draw = ImageDraw.Draw(x_pass_filter)
    outside_r = shifted_f_uv.shape[0] // (n + 2)
    inside_r = 0
    center = (shifted_f_uv.shape[0] // 2, shifted_f_uv.shape[1] // 2)
    filtered_spectrum = []
    filtered_f_xy = []
    for l in range(n - 1):
        # clear
        draw.rectangle((0, 0, shifted_f_uv.shape[0], shifted_f_uv.shape[1]), fill=0)
        # ring of outside
        ellipse_pos = (center[0] - outside_r, center[1] - outside_r, center[0] + outside_r, center[1] + outside_r)
        draw.ellipse(ellipse_pos, fill=255)
        # ring of inside
        ellipse_pos = (center[0] - inside_r, center[1] - inside_r, center[0] + inside_r, center[1] + inside_r)
        draw.ellipse(ellipse_pos, fill=0)
        # filter -> np array
        filter_array = np.asarray(x_pass_filter)
        # apply filter
        filtered_f_uv = np.multiply(shifted_f_uv, filter_array)
        # save filtered f_uv
        filtered_spectrum.append(filtered_f_uv.copy())
        # to power spectrum
        magnitude_spectrum2d = 20 * np.log(np.absolute(filtered_f_uv) + np.finfo(np.float32).eps)
        # reverse
        unshifted_f_uv = np.fft.fftshift(filtered_f_uv)
        # filtered image
        i_f_xy = np.fft.ifft2(unshifted_f_uv).real
        # save filtered f_xy
        filtered_f_xy.append(i_f_xy.copy())
        # output as image
        save_image(i_f_xy, filter_array, magnitude_spectrum2d, l)
        # add ellipse r
        outside_r += shifted_f_uv.shape[0] / (n + 2)
        inside_r += shifted_f_uv.shape[1] / (n + 2)
    # last filter
    draw.rectangle((0, 0, shifted_f_uv.shape[0], shifted_f_uv.shape[1]), fill=255)
    # inside_r += shifted_f_uv.shape[0]
    # ring
    ellipse_pos = (center[0] - inside_r, center[1] - inside_r, center[0] + inside_r, center[1] + inside_r)
    draw.ellipse(ellipse_pos, fill=0)
    # filter -> np array
    filter_array = np.asarray(x_pass_filter)
    # apply filter
    filtered_f_uv = np.multiply(shifted_f_uv, filter_array)
    # to power spectrum
    magnitude_spectrum2d = 20 * np.log(np.absolute(filtered_f_uv) + np.finfo(np.float32).eps)
    # reverse
    unshifted_f_uv = np.fft.fftshift(filtered_f_uv)
    # filtered image
    i_f_xy = np.fft.ifft2(unshifted_f_uv).real
    # output as image
    save_image(i_f_xy, filter_array, magnitude_spectrum2d, n - 1)

def modulation_gray(filename):
    # load image
    img = Image.open("./img/" + filename).convert("L")
    f_xy = np.asarray(img)

    # フーリエ変換
    f_uv = np.fft.fft2(f_xy)
    shifted_f_uv = np.fft.fftshift(f_uv)

    n = 6
    x_pass_filter = Image.new(mode="L", size=(shifted_f_uv.shape[0], shifted_f_uv.shape[1]), color=0)
    draw = ImageDraw.Draw(x_pass_filter)
    # outside_r = 40 * 2 # freq <= 40
    # inside_r = 0.6 * 3.416 * 2
    outside_r = 65 # freq <= 40
    inside_r = 5
    center = (shifted_f_uv.shape[0] // 2, shifted_f_uv.shape[1] // 2)
    # outside_r += shifted_f_uv.shape[0] / (n + 2) * 4
    # inside_r += shifted_f_uv.shape[1] / (n + 2) * 2
    # ring of outside
    ellipse_pos = (center[0] - outside_r, center[1] - outside_r, center[0] + outside_r, center[1] + outside_r)
    draw.ellipse(ellipse_pos, fill=255)
    # ring of inside
    ellipse_pos = (center[0] - inside_r, center[1] - inside_r, center[0] + inside_r, center[1] + inside_r)
    draw.ellipse(ellipse_pos, fill=0)
    # filter -> np array
    filter_array = np.asarray(x_pass_filter)

    # 保存先作成
    out_dir = "./result/multiband/" + os.path.splitext(filename)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += "/"

    scalefactor = [0.50, 0.65, 0.75, 1.0, 1.25, 1.35, 1.5]
    gray_signal = []
    shifted_f_uv_ms = shifted_f_uv.copy()
    for f in scalefactor:
        for j in range(filter_array.shape[0]):
            for i in range(filter_array.shape[1]):
                if (filter_array[j][i] == 255):
                    shifted_f_uv_ms[j][i] = shifted_f_uv[j][i] * f
                else:
                    shifted_f_uv_ms[j][i] = shifted_f_uv[j][i]
        unshifted_f_uv = np.fft.fftshift(shifted_f_uv_ms)
        i_f_xy = np.fft.ifft2(unshifted_f_uv).real
        Image.fromarray(i_f_xy).convert("L").save(out_dir + "scale_{:.2f}.png".format(f))
        gray_signal.append((Image.fromarray(i_f_xy).convert("L"), "{:.2f}".format(f)))
    Image.fromarray(f_xy).convert("L").save(out_dir + "scale_1.00.png")
    Image.fromarray(filter_array).convert("L").save(out_dir + "filter.png")

    # コントラスト感度
    filter_array_cont = np.zeros(shape=(shifted_f_uv.shape[0], shifted_f_uv.shape[1]), dtype=np.float32)
    center = (filter_array_cont.shape[0] // 2, filter_array_cont.shape[1] // 2)
    rate_set = [0.50, 0.65, 0.75, 1.25, 1.35, 1.5]
    a = 75
    b = 0.2
    c = 0.9
    K = 46
    angle = 3.416
    max_f_ratio = max_contrast_sensitivity_ratio()
    for rate in rate_set:
        for j in range(filter_array_cont.shape[1]):
            for i in range(filter_array_cont.shape[0]):
                f = math.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2) / angle
                if rate > 1.0:
                    w = (K + a * (f ** c)) * (math.e ** (-1 * b * f)) / K
                    filter_array_cont[j][i] = w
                    reg_ratio = (w - 1.0 / max_f_ratio - 1.0) * (rate - 1.0) + 1.0
                else:
                    _rate = 1.0 + (1.0 - rate)
                    # w = _rate * (K + a * (f ** c)) * (math.e ** (-1 * b * f)) / K
                    w = (K + a * (f ** c)) * (math.e ** (-1 * b * f)) / K
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
        Image.fromarray(i_f_xy).convert("L").save(out_dir + "contrast_sensitivity_scale_{:.3f}.png".format(rate))
        gray_signal.append((Image.fromarray(i_f_xy).convert("L"), "contrast_sensitivity_scale_{:.3f}".format(rate)))
        Image.fromarray(filter_array_cont * 50).convert("L").save(out_dir + "filter_contrast_scale_{:.3f}.png".format(rate))

    return gray_signal

def modulation_color(filename):
    # load image
    Y, Cb, Cr = Image.open("./img/" + filename).convert("YCbCr").split()
    f_xy = np.asarray(Y)

    # フーリエ変換
    f_uv = np.fft.fft2(f_xy)
    shifted_f_uv = np.fft.fftshift(f_uv)

    n = 6
    x_pass_filter = Image.new(mode="L", size=(shifted_f_uv.shape[0], shifted_f_uv.shape[1]), color=0)
    draw = ImageDraw.Draw(x_pass_filter)
    outside_r = 40 * 2 # freq <= 40
    inside_r = 0.6 * 3.416 * 2
    center = (shifted_f_uv.shape[0] // 2, shifted_f_uv.shape[1] // 2)
    # outside_r += shifted_f_uv.shape[0] / (n + 2) * 4
    # inside_r += shifted_f_uv.shape[1] / (n + 2) * 2
    # ring of outside
    ellipse_pos = (center[0] - outside_r, center[1] - outside_r, center[0] + outside_r, center[1] + outside_r)
    draw.ellipse(ellipse_pos, fill=255)
    # ring of inside
    ellipse_pos = (center[0] - inside_r, center[1] - inside_r, center[0] + inside_r, center[1] + inside_r)
    draw.ellipse(ellipse_pos, fill=0)
    # filter -> np array
    filter_array = np.asarray(x_pass_filter)

    # 保存先作成
    out_dir = "./result/multiband/" + os.path.splitext(filename)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += "/"

    scalefactor = [0.5, 0.75, 1.25, 1.5]
    shifted_f_uv_ms = shifted_f_uv.copy()
    for f in scalefactor:
        for j in range(filter_array.shape[0]):
            for i in range(filter_array.shape[1]):
                if (filter_array[j][i] == 255):
                    shifted_f_uv_ms[j][i] = shifted_f_uv[j][i] * f
                else:
                    shifted_f_uv_ms[j][i] = shifted_f_uv[j][i]
        unshifted_f_uv = np.fft.fftshift(shifted_f_uv_ms)
        i_f_xy = np.fft.ifft2(unshifted_f_uv).real
        i_f_xy = i_f_xy.astype(np.uint8)
        i_f_xy = Image.fromarray(i_f_xy)
        out = Image.merge("YCbCr", (i_f_xy, Cb, Cr)).convert('RGB')
        out.save(out_dir + "scale_" + str(f) + "_color.png")
    Image.merge("YCbCr", (Y, Cb, Cr)).convert("RGB").save(out_dir + "scale_1.0_color.png")

    # コントラスト感度
    filter_array_cont = np.zeros(shape=(shifted_f_uv.shape[0], shifted_f_uv.shape[1]), dtype=np.float32)
    center = (filter_array_cont.shape[0] // 2, filter_array_cont.shape[1] // 2)
    a = 75
    b = 0.2
    c = 0.9
    K = 46
    angle = 3.416
    for j in range(filter_array_cont.shape[1]):
        for i in range(filter_array_cont.shape[0]):
            f = math.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2) / angle
            w = (K + a * (f ** c)) * (math.e ** (-1 * b * f)) / K
            filter_array_cont[j][i] = w
            if filter_array_cont[j][i] >= 1.0:
                shifted_f_uv_ms[j][i] = shifted_f_uv[j][i] * w
            else:
                shifted_f_uv_ms[j][i] = shifted_f_uv[j][i]
    unshifted_f_uv = np.fft.fftshift(shifted_f_uv_ms)
    i_f_xy = np.fft.ifft2(unshifted_f_uv).real
    i_f_xy = i_f_xy.astype(np.uint8)
    i_f_xy = np.clip(i_f_xy, 0, 255)
    i_f_xy = Image.fromarray(i_f_xy)
    Image.merge("YCbCr", (i_f_xy, Cb, Cr)).convert("RGB").save(out_dir + "contrast_sensitivity_color.png")

def modulation_color_using_gray(filename, gray, org) -> None:
    color_space = "YCbCr"
    # load image
    img = Image.open("./img/" + filename).convert("L")
    org = Image.open("./img/" + org).convert('HSV')
    yy, cb, cr = Image.open("./img/" + filename).convert(color_space).split()
    _, S, _ = org.split()
    S = np.asarray(S)
    sat_mean = S.mean()
    # 保存先作成
    out_dir = "./result/multiband/" + os.path.splitext(filename)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += "/"
    for (signal, string) in gray:
        out = Image.merge(color_space, (signal, cb, cr)).convert("RGB")
        out.save(out_dir + string + "_color.png")
        # addition
        SaturationEnhance = ImageEnhance.Color(out)
        _, sat, _ = out.convert('HSV').split()
        sat = np.asarray(sat)
        SaturationEnhance = SaturationEnhance.enhance(math.pow(sat_mean / sat.mean(), 1))
        # SaturationEnhance = SaturationEnhance.enhance(0.85)
        SaturationEnhance.convert('RGB').save(out_dir + string + "_color_AS.png")

def adjust_average_luminance(filename, gray):
    color_space = "YCbCr"
    # load image
    img = Image.open("./img/" + filename).convert("L")
    img = np.asarray(img)
    yy, cb, cr = Image.open("./img/" + filename).convert(color_space).split()
    # 保存先
    out_dir = "./result/multiband/" + os.path.splitext(filename)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += "/"
    # 原画像の平均輝度
    org_ave_liminance = int(img.mean())
    # 変調後の平均輝度を原画像の平均輝度に合わせる
    # 平均輝度を調整後の信号を色差信号と組み合わせる
    print("org ave liminance: ", org_ave_liminance)
    for (signal, string) in gray:
        signal = np.asarray(signal, dtype=np.uint8)
        modulated_ave_liminance = int(signal.mean())
        diff = modulated_ave_liminance - org_ave_liminance
        adjusted_signal = signal - diff
        print("before ave luminance: ", modulated_ave_liminance)
        print("diff: ", diff)
        print("adjusted ave luminance: ", int(adjusted_signal.mean()))
        adjusted_signal = adjusted_signal.astype(np.uint8)
        Image.merge(color_space, (Image.fromarray(adjusted_signal), cb, cr)).convert("RGB").save(out_dir + string + "_color_adjust.png")

if __name__ == '__main__':
    # multiscale("./img/tex001.bmp")
    # modulation("./img/tex001.bmp")

    files = os.listdir("./img")
    # files = ["bird.png", "flower.png", "table.png", "tex009.bmp"]
    # files = ["flower.bmp"]
    for file in files:
        print(file)
        gray = modulation_gray(file)
        # modulation_color(file)
        # tmp(file)
        modulation_color_using_gray(file, gray, file)
        # adjust_average_luminance(file, gray)