import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="画像カラー分析システム",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 画像カラー分析システム")

st.markdown("""
### ようこそ！

このアプリケーションは、画像を分析するためのシステムです。

画像をアップロードすると、以下の情報を確認できます。

- 🖼️ 元画像
- ⚫ グレースケール画像
- 📐 エッジ検出
- 📊 RGBヒストグラム
- 🌈 HSVカラー分析
- 📈 画像情報
- 🎨 平均RGBカラー
- 💾 エッジ画像のダウンロード


""")

uploaded = st.file_uploader(
    "📂 画像をアップロードしてください",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🖼 元画像")
        st.image(img, use_container_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    with col2:
        st.subheader("⚫ グレースケール")
        st.image(gray, use_container_width=True)

    edges = cv2.Canny(gray, 100, 200)

    with col3:
        st.subheader("📐 エッジ検出")
        st.image(edges, use_container_width=True)

    st.divider()

    st.subheader("📊 RGBヒストグラム")

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = ("red", "green", "blue")

    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)

    ax.set_xlim([0, 256])
    ax.set_xlabel("画素値")
    ax.set_ylabel("頻度")
    ax.grid()

    st.pyplot(fig)

    st.divider()

    st.subheader("📈 画像情報")

    h, w, c = img.shape
    avg = img.mean(axis=(0, 1))

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("幅", w)
    c2.metric("高さ", h)
    c3.metric("チャンネル数", c)
    c4.metric("総画素数", f"{w*h:,}")

    st.divider()

    st.subheader("🎨 平均RGBカラー")

    r, g, b = avg

    c1, c2, c3 = st.columns(3)

    c1.metric("赤", f"{r:.2f}")
    c2.metric("緑", f"{g:.2f}")
    c3.metric("青", f"{b:.2f}")

    color = np.zeros((150, 400, 3), dtype=np.uint8)
    color[:] = avg.astype(np.uint8)

    st.image(color, caption="平均カラー")

    st.divider()

    st.subheader("🌈 HSVカラー分析")

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])

    st.write(f"**色相（Hue）:** {h_mean:.2f}")
    st.write(f"**彩度（Saturation）:** {s_mean:.2f}")
    st.write(f"**明るさ（Brightness）:** {v_mean:.2f}")

    st.progress(min(v_mean / 255, 1.0))

    st.divider()

    st.subheader("💾 エッジ画像のダウンロード")

    edge_png = cv2.imencode(".png", edges)[1].tobytes()

    st.download_button(
        "エッジ画像をダウンロード",
        edge_png,
        "edge_image.png",
        "image/png"
    )

else:
    st.info("⬆ 画像をアップロードしてください。")