import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SEC-Assistant", page_icon="🏦", layout="wide")

st.title("🏦 SEC-Assistant: Financial Q&A")
st.markdown("Fine-tuned Phi-3-Mini for SEC filing Q&A with citation grounding")
st.markdown("[🤗 Model](https://huggingface.co/Abrar144/sec-assistant-phi3-finqa) | [💻 GitHub](https://github.com/Abrar144/sec-assistant-finqa)")
st.divider()

@st.cache_data
def load_data():
    with open("all_results.json") as f:
        return json.load(f)

data = load_data()
m = data["metrics"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Metrics", "🔍 Before vs After", "🧪 Forgetting", "🏦 Demo", "📈 Training"])

with tab1:
    st.header("Performance Comparison")
    c1, c2, c3 = st.columns(3)
    c1.metric("Citation Rate", f"{m['citation_ft']}%", f"+{m['citation_ft']-m['citation_base']}%")
    c2.metric("Answer Rate", f"{100-m['idk_ft']}%", f"+{m['idk_base']-m['idk_ft']}%")
    c3.metric("Numeric Accuracy", f"{m['numeric_ft']}%", f"+{m['numeric_ft']-m['numeric_base']}%")
    fig = px.bar(
        pd.DataFrame({
            "Metric": ["Citation", "Answer Rate", "Numeric"] * 2,
            "Score": [m['citation_base'], 100-m['idk_base'], m['numeric_base'], m['citation_ft'], 100-m['idk_ft'], m['numeric_ft']],
            "Model": ["Base"]*3 + ["Fine-tuned"]*3
        }),
        x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_map={"Base": "#FF6B6B", "Fine-tuned": "#51CF66"}
    )
    st.plotly_chart(fig, use_container_width=True)
    radar = go.Figure()
    cats = ["Citation", "Answer Rate", "Numeric"]
    radar.add_trace(go.Scatterpolar(r=[m['citation_base'], 100-m['idk_base'], m['numeric_base']], theta=cats, fill='toself', name='Base', line_color='red', opacity=0.5))
    radar.add_trace(go.Scatterpolar(r=[m['citation_ft'], 100-m['idk_ft'], m['numeric_ft']], theta=cats, fill='toself', name='Fine-tuned', line_color='green', opacity=0.5))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), height=400)
    st.plotly_chart(radar, use_container_width=True)

with tab2:
    st.header("Side-by-Side Examples")
    for i, c in enumerate(data["comparison"]):
        with st.expander(f"Q{i+1}: {c['question'][:80]}"):
            st.markdown(f"**True:** `{c['true_answer']}`")
            c1, c2 = st.columns(2)
            c1.markdown("### Base Model")
            c1.error(c["base_answer"][:300])
            c2.markdown("### Fine-tuned")
            c2.success(c["finetuned_answer"][:300])

with tab3:
    st.header("Catastrophic Forgetting Check")
    p = sum(1 for r in data["forgetting"] if r["preserved"])
    st.success(f"✅ {p}/{len(data['forgetting'])} general capabilities preserved")
    for r in data["forgetting"]:
        st.markdown(f"{'✅' if r['preserved'] else '❌'} **[{r['category']}]** {r['question']}")

with tab4:
    st.header("SEC Filing Demo (Apple 10-K)")
    for d in data["demo"]:
        with st.expander(f"❓ {d['question']}"):
            st.info(f"Type: {d['type']}")
            st.success(d["answer"])

with tab5:
    st.header("Training Progress")
    t = data["training"]
    if t.get("steps"):
        st.plotly_chart(px.line(x=t["steps"], y=t["losses"], labels={"x":"Step","y":"Loss"}, title="Loss Curve"), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Start Loss", f"{t['start_loss']:.4f}")
    c2.metric("End Loss", f"{t['end_loss']:.4f}")
    c3.metric("Reduction", f"{t['reduction_pct']}%")
    st.markdown(f"Training: {t['train_examples']} examples ({t['normal_count']} normal + {t['idk_count']} IDK)")

st.divider()
st.markdown("**Built with:** Phi-3-Mini | QLoRA | FinQA | Kaggle T4")