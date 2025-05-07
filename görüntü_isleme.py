from PIL import Image, ImageFilter
import random
import time
import math
import os
import matplotlib.pyplot as plt
import numpy as np

def contrast_arttirma(image, factor):
    width, height = image.size
    pixels = image.load()

    enhanced_image = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            r = int((r - 128) * factor + 128)
            g = int((g - 128) * factor + 128)
            b = int((b - 128) * factor + 128)
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            enhanced_image.putpixel((x, y), (r, g, b))

    return enhanced_image

def resim_bolme(image1, image2):
    width, height = image1.size
    image2 = image2.resize((width, height))

    new_image = Image.new("RGB", (width, height))
    pixels1 = image1.load()
    pixels2 = image2.load()

    for x in range(width):
        for y in range(height):
            r1, g1, b1 = pixels1[x, y]
            r2, g2, b2 = pixels2[x, y]
            r_div = int(r1 / (r2 if r2 != 0 else 1))
            g_div = int(g1 / (g2 if g2 != 0 else 1))
            b_div = int(b1 / (b2 if b2 != 0 else 1))
            new_image.putpixel((x, y), (r_div, g_div, b_div))

    return new_image

def resim_ekleme(image1, image2):
    width, height = image1.size
    image2 = image2.resize((width, height))

    new_image = Image.new("RGB", (width, height))
    pixels1 = image1.load()
    pixels2 = image2.load()

    for x in range(width):
        for y in range(height):
            r1, g1, b1 = pixels1[x, y]
            r2, g2, b2 = pixels2[x, y]
            r_sum = min(r1 + r2, 255)
            g_sum = min(g1 + g2, 255)
            b_sum = min(b1 + b2, 255)
            new_image.putpixel((x, y), (r_sum, g_sum, b_sum))

    return new_image

def renk_uzay_donusumu(image):
    width, height = image.size
    grayscale_image = Image.new("L", (width, height))

    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            gray_value = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
            grayscale_image.putpixel((x, y), gray_value)

    return grayscale_image

def resim_kirpma(image, left, upper, right, lower):
    width, height = image.size

    cropped_image = Image.new("RGB", (right - left, lower - upper))

    pixels = image.load()

    for x in range(left, right):
        for y in range(upper, lower):
            if 0 <= x < width and 0 <= y < height:
                cropped_image.putpixel((x - left, y - upper), pixels[x, y])

    return cropped_image

def histogram_germe(image):
    grayscale = renk_uzay_donusumu(image)
    width, height = grayscale.size
    pixels = grayscale.load()

    # Min ve max parlaklık değerlerini bul
    min_val = 255
    max_val = 0
    for x in range(width):
        for y in range(height):
            p = pixels[x, y]
            if p < min_val:
                min_val = p
            if p > max_val:
                max_val = p

    # Yeni görüntü oluştur
    stretched_image = Image.new("L", (width, height))

    for x in range(width):
        for y in range(height):
            old_pixel = pixels[x, y]
            if max_val - min_val == 0:
                new_pixel = 0
            else:
                new_pixel = int((old_pixel - min_val) * 255 / (max_val - min_val))
            stretched_image.putpixel((x, y), new_pixel)

    return stretched_image

def histogram_genisletme(image, alt=100, ust=175):
    # Görüntüyü gri seviyeye çevir
    image = image.convert("L")
    pixels = np.array(image)

    # Belirtilen aralığa yayma işlemi
    expanded = ((pixels - pixels.min()) / (pixels.max() - pixels.min()) * (ust - alt) + alt).astype(np.uint8)
    return Image.fromarray(expanded)

def histogram_grafigi_goster(orijinal, islemli, baslik1, baslik2):
    # Histogramları al
    hist_orig = orijinal.histogram()
    hist_new = islemli.histogram()

    # Grafik çiz
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(hist_orig, color='gray')
    plt.title(baslik1)
    plt.xlabel("Piksel Değeri")
    plt.ylabel("Piksel Sayısı")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(hist_new, color='blue')
    plt.title(baslik2)
    plt.xlabel("Piksel Değeri")
    plt.ylabel("Piksel Sayısı")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def uzaklastir(image, scale_factor):
    width, height = image.size
    new_width = int(width / scale_factor)
    new_height = int(height / scale_factor)

    if new_width < 1 or new_height < 1:
        raise ValueError("Uzaklaştırma oranı çok yüksek, görüntü yok oluyor.")

    resized_image = Image.new("RGB", (new_width, new_height))
    pixels = image.load()

    for x in range(new_width):
        for y in range(new_height):
            src_x = int(x * scale_factor)
            src_y = int(y * scale_factor)
            src_x = min(src_x, width - 1)
            src_y = min(src_y, height - 1)
            resized_image.putpixel((x, y), pixels[src_x, src_y])

    return resized_image

def yakinlastir(image, yakinlastirma_derecesi):
    if yakinlastirma_derecesi <= 1:
        raise ValueError("Yakınlaştırma için ölçek oranı 1'den büyük olmalıdır.")

    width, height = image.size
    new_width = int(width / yakinlastirma_derecesi)
    new_height = int(height / yakinlastirma_derecesi)

    center_x = width // 2
    center_y = height // 2

    left = max(center_x - new_width // 2, 0)
    upper = max(center_y - new_height // 2, 0)
    right = min(center_x + new_width // 2, width)
    lower = min(center_y + new_height // 2, height)

    cropped_image = image.crop((left, upper, right, lower))

    zoomed_image = cropped_image.resize((width, height), Image.NEAREST)

    return zoomed_image

def gri_yap(image):
    width, height = image.size
    pixels = image.load()

    gray_image = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            gri = int((r + g + b) / 3)
            gray_image.putpixel((x, y), (gri, gri, gri))
    return gray_image

def binary_donusum(image, esik=128):
    width, height = image.size
    pixels = image.load()

    bw_image = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            ortalama = (r + g + b) // 3
            deger = 255 if ortalama >= esik else 0
            bw_image.putpixel((x, y), (deger, deger, deger))
    return bw_image

def resim_dondur(image_path, angle):

    try:
        # Görüntüyü aç
        image = Image.open(image_path)

        # Görüntüyü döndür
        rotated_image = image.rotate(angle, expand=True)  # expand=True, yeni boyutu otomatik ayarlamak için kullanılır

        return rotated_image

    except Exception as e:
        print(f"Görüntü işlenirken bir hata oluştu: {str(e)}")
        return None

def salt_pepper_ekleme(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = image.copy()
    pixels = noisy_image.load()
    width, height = noisy_image.size

    for x in range(width):
        for y in range(height):
            rand = random.random()
            if rand < salt_prob:
                pixels[x, y] = 255  # tuz
            elif rand > 1 - pepper_prob:
                pixels[x, y] = 0  # karabiber
    return noisy_image

def mean_filter(image):
    filtered = Image.new("L", image.size)
    pixels = image.load()
    out = filtered.load()
    width, height = image.size

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            total = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    total += pixels[x + i, y + j]
            out[x, y] = total // 9
    return filtered

def median_filter(image):
    filtered = Image.new("L", image.size)
    pixels = image.load()
    out = filtered.load()
    width, height = image.size

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            values = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    values.append(pixels[x + i, y + j])
            values.sort()
            out[x, y] = values[4]
    return filtered

def unsharp(image, strength=5, blur_radius=6):
    if image.mode != "L":
        image = image.convert("L")

    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    result = Image.new("L", image.size)

    orig_pixels = image.load()
    blur_pixels = blurred.load()
    out_pixels = result.load()

    width, height = image.size
    for x in range(width):
        for y in range(height):
            sharpened = orig_pixels[x, y] + strength * (orig_pixels[x, y] - blur_pixels[x, y])
            out_pixels[x, y] = max(0, min(255, int(sharpened)))
    return result

def dilation(image):
    image = image.convert("L")
    result = Image.new("L", image.size)
    pixels = image.load()
    out = result.load()
    width, height = image.size

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            max_val = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    max_val = max(max_val, pixels[x + i, y + j])
            out[x, y] = max_val
    return result

def erosion(image):
    image = image.convert("L")
    result = Image.new("L", image.size)
    pixels = image.load()
    out = result.load()
    width, height = image.size

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            min_val = 255
            for i in range(-1, 2):
                for j in range(-1, 2):
                    min_val = min(min_val, pixels[x + i, y + j])
            out[x, y] = min_val
    return result

def opening(image):
    return dilation(erosion(image))

def closing(image):
    return erosion(dilation(image))

def konvolusyon_islemi(image, kernel_size=3):
    start_time = time.time()
    width, height = image.size

    output_image = Image.new('RGB', (width, height))
    pad = kernel_size // 2
    update_interval = max(1, height // 10)
    last_update = 0

    for y in range(height):
        if y % update_interval == 0:
            percent_done = (y * 100) // height
            if percent_done > last_update:
                last_update = percent_done

        for x in range(width):
            r_total = g_total = b_total = 0
            count = 0

            for j in range(-pad, pad + 1):
                for i in range(-pad, pad + 1):
                    if 0 <= x + i < width and 0 <= y + j < height:
                        r, g, b = image.getpixel((x + i, y + j))
                        r_total += r
                        g_total += g
                        b_total += b
                        count += 1

            if count > 0:
                r_avg = r_total // count
                g_avg = g_total // count
                b_avg = b_total // count
            else:
                r_avg, g_avg, b_avg = image.getpixel((x, y))

            r_avg = max(0, min(255, r_avg))
            g_avg = max(0, min(255, g_avg))
            b_avg = max(0, min(255, b_avg))

            output_image.putpixel((x, y), (r_avg, g_avg, b_avg))

    elapsed_time = time.time() - start_time
    print(f"İşlem tamamlandı! Geçen süre: {elapsed_time:.2f} saniye")

    return output_image

def kenar_bulma_prewit(image_path, threshold=30):
    start_time = time.time()
    print(f"Prewitt kenar bulma uygulanıyor (eşik değeri: {threshold})...")

    try:
        # Görüntüyü yükle ve RGB moduna dönüştür
        image = Image.open(image_path).convert("RGB")
        # Gri seviyeye dönüştür
        gray_image = image.convert("L")
        width, height = gray_image.size

        # Yeni bir görüntü oluştur
        output_image = Image.new('RGB', (width, height))

        # Prewitt operatörleri (yatay ve dikey)
        prewitt_x = [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]

        prewitt_y = [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ]

        # Görüntüyü piksel dizisine dönüştür
        pixel_data = list(gray_image.getdata())
        pixels = [pixel_data[i * width:(i + 1) * width] for i in range(height)]

        # Her piksel için işlem yap (kenar pikselleri hariç)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Yatay ve dikey gradyan hesapla
                gradient_x = 0
                gradient_y = 0

                # 3x3 komşuluk için Prewitt operatörlerini uygula
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        pixel_val = pixels[y + i][x + j]
                        gradient_x += pixel_val * prewitt_x[i + 1][j + 1]
                        gradient_y += pixel_val * prewitt_y[i + 1][j + 1]

                # Gradyan büyüklüğünü hesapla
                gradient_magnitude = math.sqrt(gradient_x ** 2 + gradient_y ** 2)

                # Eşik değerine göre kenar belirle
                if gradient_magnitude > threshold:
                    # Kenar piksel (beyaz)
                    output_image.putpixel((x, y), (255, 255, 255))
                else:
                    # Kenar olmayan piksel (siyah)
                    output_image.putpixel((x, y), (0, 0, 0))

        # İşlem tamamlandı
        elapsed_time = time.time() - start_time
        print(f"İşlem tamamlandı! Geçen süre: {elapsed_time:.2f} saniye")

        return output_image

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Kenar bulma hatası: {str(e)}")
        return None

def esikleme_islemleri(image_path, threshold_value=128):
    start_time = time.time()
    print(f"Tek Eşikleme uygulanıyor (eşik değeri: {threshold_value})...")

    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {image_path}")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        output_image = Image.new('RGB', (width, height))

        update_interval = max(1, height // 10)
        last_update = 0

        print(f"İşleniyor: Görüntü boyutu {width}x{height} piksel")

        for y in range(height):
            if y % update_interval == 0:
                percent_done = (y * 100) // height
                if percent_done > last_update:
                    last_update = percent_done
                    print(f"İşleniyor: %{percent_done} tamamlandı")

            for x in range(width):
                try:
                    r, g, b = image.getpixel((x, y))
                    gray_value = (r + g + b) // 3

                    if gray_value >= threshold_value:
                        new_color = (255, 255, 255)  # Beyaz
                    else:
                        new_color = (0, 0, 0)  # Siyah

                    output_image.putpixel((x, y), new_color)
                except Exception as e:
                    print(f"Piksel işleme hatası ({x},{y}): {str(e)}")
                    continue

        elapsed_time = time.time() - start_time
        print(f"İşlem tamamlandı! Geçen süre: {elapsed_time:.2f} saniye")

        return output_image

    except FileNotFoundError as fnf_error:
        print(f"Dosya hatası: {fnf_error}")
        return None
    except Exception as e:
        print(f"Eşikleme hatası: {str(e)}")
        return None

def histogram_germe(image):
    # Görüntüyü gri seviyeye çevir
    image = image.convert("L")
    pixels = np.array(image)

    min_val = pixels.min()
    max_val = pixels.max()

    # Germe işlemi: 0-255 aralığına ölçekle
    stretched = ((pixels - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return Image.fromarray(stretched)


image = Image.open("deneme.jpg")
denemee = resim_kirpma(image, 50, 100, 150, 160).show()
