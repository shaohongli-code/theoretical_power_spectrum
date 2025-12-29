# theoretical_power_spectrum


[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://python.org)

这是一个用于计算和分析信号/数据**功率谱密度 (PSD)** 的高效工具库。它支持多种窗函数、去趋势处理以及多维数据的频谱分析。

## 🚀 功能特性

* **多种算法支持**：包括周期图法 (Periodogram)、韦尔奇法 (Welch's method) 等。
* **多维计算**：支持一维时间序列、二维图像及三维空间的功率谱分析。
* **可视化**：内置绘图函数，一键生成频谱图（Log-Log 坐标）。
* **高性能**：底层采用 NumPy/FFTW 优化，处理大规模数据速度快。

## 📦 安装说明

你可以通过 pip 直接安装：

```bash
git clone git@github.com:shaohongli-code/theoretical_power_spectrum.git
pip install ./

