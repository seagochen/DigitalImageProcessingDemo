from Utilities.DiagnoseTool import calculate_hist
from Utilities.DiagnoseTool import load_image_gray
from Utilities.DiagramPlotter import DiagramPlotter


def analysis_image_hists(filepath: str):
    img = load_image_gray(filepath)
    img_hist = calculate_hist(img)
    return img, img_hist


def spatial_noise_analysis():

    img_1, hist_1 = analysis_image_hists("./Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/"
                                         "Fig0503 (original_pattern).tif")
    img_2, hist_2 = analysis_image_hists("./Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/"
                                         "Fig0504(a)(gaussian-noise).tif")
    img_3, hist_3 = analysis_image_hists("./Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/"
                                         "Fig0504(b)(rayleigh-noise).tif")
    img_4, hist_4 = analysis_image_hists("./Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/"
                                         "Fig0504(c)(gamma-noise).tif")
    img_5, hist_5 = analysis_image_hists("./Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/"
                                         "Fig0504(g)(neg-exp-noise).tif")
    img_6, hist_6 = analysis_image_hists("./Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/"
                                         "Fig0504(h)(uniform-noise).tif")
    img_7, hist_7 = analysis_image_hists("./Data/DIP/DIP3E_CH05_Original_Images/DIP3E_CH05_Original_Images/"
                                         "Fig0504(i)(salt-pepper-noise).tif")
    chart = DiagramPlotter()
    chart.append_image(img_1, "original_pattern").append_bars(hist_1, "original_pattern")
    chart.append_image(img_2, "gaussian-noise").append_bars(hist_2, "gaussian-noise")
    chart.append_image(img_3, "rayleigh-noise").append_bars(hist_3, "rayleigh-noise")
    chart.show(3, 2)

    chart.clean()
    chart.append_image(img_4, "gamma-noise").append_bars(hist_4, "gamma-noise")
    chart.append_image(img_5, "neg-exp-noise").append_bars(hist_5, "neg-exp-noise")
    chart.append_image(img_6, "uniform-noise").append_bars(hist_6, "uniform-noise")
    chart.append_image(img_7, "salt-pepper-noise").append_bars(hist_7, "salt-pepper-noise")
    chart.show(4, 2)
