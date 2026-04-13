from PIL import Image
import streamlit as st

from src.inference_utils import load_class_metrics, predict_pil_image

st.set_page_config(page_title="Animal Image Classifier", page_icon="🐾")

st.title("Animal Image Classifier")
st.write("Upload an image and the model will predict the animal with confidence scores.")

uploaded_file = st.file_uploader(
    "Upload an animal image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption=uploaded_file.name, use_container_width=True)

    try:
        predictions = predict_pil_image(image=image, top_k=3)
        class_metrics = load_class_metrics()

        top_label, top_prob = predictions[0]

        st.subheader("Prediction")
        st.write(f"**Animal:** {top_label}")
        st.write(f"**Confidence:** {top_prob:.2%}")

        if class_metrics and top_label in class_metrics:
            m = class_metrics[top_label]
            st.write(f"**Precision:** {m['precision']:.3f}  |  **Recall:** {m['recall']:.3f}  |  **F1 Score:** {m['f1']:.3f}")

        st.subheader("Top Predictions")
        for label, prob in predictions:
            metric_str = ""
            if class_metrics and label in class_metrics:
                m = class_metrics[label]
                metric_str = (
                    f" | P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
                )
            st.write(f"- **{label}**: {prob:.2%}{metric_str}")

        if top_prob < 0.50:
            st.warning("Low confidence prediction. The image may be unclear or outside the trained classes.")

    except FileNotFoundError:
        st.error("No trained model found. Run prepare_data.py, train.py, and evaluate.py first.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
