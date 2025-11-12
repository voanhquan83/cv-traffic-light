import os, json, tempfile
import numpy as np
import cv2
import streamlit as st

# ---- import detector (đặt cùng thư mục) ----
from traffic_light_detection import detect_traffic_light_color, letterbox_resize
st.set_page_config(page_title="Nhận diện đèn giao thông (local)", page_icon="🚦", layout="centered")

st.title("🚦 Nhận diện giá trị đèn giao thông")
st.caption("Chọn 1 ảnh → phân tích màu đèn đang sáng, kèm kết luận tiếng Việt")

# Tuỳ chọn
col1, col2 = st.columns(2)
force_orientation = col1.selectbox("Hướng đèn", ["Tự phát hiện", "Ép đèn dọc", "Ép đèn ngang"])
denoise = col2.selectbox("Bộ lọc nhiễu", ["bilateral", "nlmeans"])

uploaded = st.file_uploader("Chọn ảnh (jpg/png/webp/bmp)", type=["jpg","jpeg","png","bmp","webp"])

def vietnamese_conclusion(result: dict) -> str:
    label_vi = {"red":"Đèn ĐỎ đang sáng", "yellow":"Đèn VÀNG đang sáng", "green":"Đèn XANH đang sáng", "unknown":"Không xác định được màu đèn"}
    ori_vi = {"vertical":"đèn dọc", "horizontal":"đèn ngang"}
    lamps = result.get("lamps", [])
    parts=[]
    for lp in lamps:
        slot=lp.get("slot","")
        slot_vi={"top":"bóng TRÊN","mid":"bóng GIỮA","bot":"bóng DƯỚI","left":"bóng TRÁI","center":"bóng GIỮA","right":"bóng PHẢI"}.get(slot,slot)
        color_vi={"red":"đỏ","yellow":"vàng","green":"xanh","unknown":"không rõ"}.get(lp.get("label","unknown"),"không rõ")
        parts.append(f"{slot_vi}: {color_vi}")
    lamps_text = "; ".join(parts) if parts else "Không phát hiện đủ 3 bóng."
    return f"{label_vi.get(result.get('label','unknown'),'Không xác định')}. Hướng: {ori_vi.get(result.get('orientation',''),'không rõ')}. Trạng thái các bóng: {lamps_text}."

def draw_vis(src_path: str, result: dict) -> np.ndarray:
    img = cv2.imread(src_path)
    vis, _, _ = letterbox_resize(img, (512, 512))
    if result.get("box"):
        x,y,w,h = result["box"]
        cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,255),2)
    text = f"{result.get('label','unknown')} | {result.get('orientation','?')} ({int(result.get('score',0))})"
    cv2.putText(vis, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)
    for lp in result.get("lamps", [])[:3]:
        bx,by,bw,bh = lp["box"]
        cv2.rectangle(vis,(bx,by),(bx+bw,by+bh),(0,255,0),2)
        tag = f"{lp['slot']}:{lp['label']}"
        cv2.putText(vis, tag, (bx, max(0,by-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1,cv2.LINE_AA)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis

if uploaded is not None:
    # Lưu tạm để OpenCV đọc
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.image(uploaded, caption="Ảnh đã chọn", use_container_width=True)

    # Chạy detect
    result = detect_traffic_light_color(tmp_path, denoise=denoise)
    if force_orientation == "Ép đèn dọc":
        result["orientation"] = "vertical"
    elif force_orientation == "Ép đèn ngang":
        result["orientation"] = "horizontal"

    # Kết luận TV
    st.subheader("Kết luận (Tiếng Việt)")
    st.success(vietnamese_conclusion(result))

    # Ảnh đã đánh dấu
    st.subheader("Ảnh đã đánh dấu")
    st.image(draw_vis(tmp_path, result), use_container_width=True)

    # JSON kết quả
    with st.expander("Xem JSON kết quả"):
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
