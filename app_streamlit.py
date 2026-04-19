import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Career Prediction UI", layout="wide")

clf_model = joblib.load('placement_classification.pkl')
reg_model = joblib.load('salary_regression.pkl')


st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("""Aplikasi ini bertindak sebagai web untuk prediksi placement status dan gaji secara langsung (Monolithic).""")
st.sidebar.markdown("---")
st.sidebar.write("**Petunjuk:** Isi form, lalu sistem akan memproses prediksi.")

st.title("Prediction Dashboard: Placement & Salary")
st.markdown("Masukkan data akademik dan skill Anda untuk melihat prospek karir masa depan.")

tab1, tab2 = st.tabs(["Data Akademik", "Profil & Keahlian"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        ssc_p = st.slider("SSC % (10th)", 0, 100, 75)
        hsc_p = st.slider("HSC % (12th)", 0, 100, 70)
        degree_p = st.slider("Degree %", 0, 100, 72)
    with col2:
        cgpa = st.number_input("Current CGPA (0-10)", 0.0, 10.0, 8.0, step=0.1)
        attendance = st.slider("Attendance %", 0, 100, 85)
        entrance_score = st.slider("Entrance Score", 0, 100, 70)

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        extracurricular = st.radio("Extracurricular", ["Yes", "No"], horizontal=True)
        st.markdown("<br>", unsafe_allow_html=True)
        tech_score = st.slider("Technical Skill Score", 0, 100, 75)
        soft_score = st.slider("Soft Skill Score", 0, 100, 80)
    with col4:
        internships = st.number_input("Internships", 0, 10, 1)
        projects = st.number_input("Live Projects", 0, 10, 2)
        certs = st.number_input("Certifications", 0, 20, 2)
        work_exp = st.number_input("Work Experience (Months)", 0, 120, 0, step=6)

ac_idx = (ssc_p + hsc_p + degree_p + (cgpa * 10)) / 4
job_ready = internships + projects + certs
comp_score = tech_score + soft_score

st.subheader("Live Profiling Insights")
col_dash1, col_dash2, col_dash3 = st.columns([1, 1, 1.5])

with col_dash1:
    fig_ac = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ac_idx,
        title={'text': "Academic Index"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00B4D8"}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray"}
    ))
    fig_ac.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_ac, use_container_width=True)

with col_dash2:
    fig_comp = go.Figure(go.Indicator(
        mode="gauge+number",
        value=comp_score,
        title={'text': "Total Competency"},
        gauge={'axis': {'range': [0, 200]}, 'bar': {'color': "#FF5A5F"}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray"}
    ))
    fig_comp.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_comp, use_container_width=True)
    st.info(f"**Job Readiness:** {job_ready} Pts")

with col_dash3:
    categories = ['Technical', 'Soft Skill', 'SSC', 'HSC', 'Degree']
    values = [tech_score, soft_score, ssc_p, hsc_p, degree_p]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Kandidat', line_color='#00B4D8'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Sebaran Kompetensi", margin=dict(l=20, r=20, t=40, b=20), height=300)
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

submit_button = st.button("Analisa Profil Saya", type="primary", use_container_width=True)

if submit_button:
    input_data = pd.DataFrame([{
        "gender": gender,
        "extracurricular_activities": extracurricular,
        "ssc_percentage": float(ssc_p),
        "hsc_percentage": float(hsc_p),
        "degree_percentage": float(degree_p),
        "cgpa": float(cgpa),
        "entrance_exam_score": float(entrance_score),
        "technical_skill_score": float(tech_score),
        "soft_skill_score": float(soft_score),
        "internship_count": int(internships),
        "live_projects": int(projects),
        "work_experience_months": int(work_exp),
        "certifications": int(certs),
        "attendance_percentage": float(attendance)
    }])

    st.markdown("---")
    
    with st.spinner("Memproses prediksi AI..."):
        try:
            placement_pred = clf_model.predict(input_data)[0]
            
            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                if placement_pred == 1 or placement_pred == "Placed":
                    st.success("**Status Prediksi: PLACED (Lulus Penempatan)**")
                    
                    salary_pred = reg_model.predict(input_data)[0]
                    final_salary = round(float(max(0, salary_pred)), 2)
                    st.metric(label="Estimasi Paket Gaji", value=f"{final_salary} LPA")
                    st.balloons()
                else:
                    st.error("**Status Prediksi: NOT PLACED**")
                    st.metric(label="Estimasi Paket Gaji", value="0.00 LPA")
            
            with col_res2:
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', line_color='blue'))
                fig_res.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Radar Profil Kompetensi", margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_res, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error saat memproses model: {e}")
