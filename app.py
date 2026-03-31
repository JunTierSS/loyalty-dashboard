"""
Loyalty Intelligence System — Dashboard Presentacion
50K clientes reales | Filtros globales | 8 vistas
streamlit run dashboard_presentacion.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyarrow.parquet as pq, pyarrow as pa
import warnings, os
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Loyalty Intelligence", page_icon="🎯", layout="wide")

# ── Load data ────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Cargando datos...")
def load_data():
    def read_pq(path):
        t = pq.read_table(path)
        cols = []
        for i in range(t.num_columns):
            col = t.column(i)
            if 'date' in str(col.type).lower(): col = col.cast(pa.string())
            cols.append(col)
        df = pa.table(cols, names=t.column_names).to_pandas()
        for c in df.select_dtypes('object').columns:
            try:
                if df[c].dropna().head(5).str.match(r'^\d{4}-').any(): df[c]=pd.to_datetime(df[c])
            except: pass
        return df

    scored = read_pq('data_scored.parquet')
    snap = read_pq('data_snapshot.parquet')

    # Fix types
    for df in [scored, snap]:
        for c in df.columns:
            if df[c].dtype == 'object' and c not in ['cust_id','tier','gender','city','dominant_retailer','funnel_state_at_t0','status','cluster_name','prioridad','canal','timing','objetivo','accion']:
                try: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                except: pass

    # Add tipo_cliente to scored
    if 'stock_points_at_t0' in scored.columns and 'y' in scored.columns:
        scored['tipo_cliente'] = np.where(scored.y>=1, 'CANJEADOR',
            np.where(scored.stock_points_at_t0>=1000, 'POTENCIAL', 'ACUMULADOR'))

    # Add quintil if missing
    if 'quintil_gasto' not in scored.columns and 'monetary_total' in scored.columns:
        scored['quintil_gasto'] = pd.qcut(scored.monetary_total.rank(method='first'), 5, labels=['Q1','Q2','Q3','Q4','Q5'])

    # Renombrar tiers
    tier_map = {'NORMAL':'Entrada','Normal':'Entrada','FAN':'Fan','Fan':'Fan','PREMIUM':'Premium','Premium':'Premium','ELITE':'Elite','Elite':'Elite'}
    for df in [scored, snap]:
        if 'tier' in df.columns:
            df['tier'] = df['tier'].map(lambda x: tier_map.get(x, x))

    # Renombrar funnel states
    funnel_map = {
        'INSCRITO':'Inscrito','PARTICIPANTE':'Participante',
        'POSIBILIDAD_CANJE':'Posibilidad Canje','CANJEADOR':'Canjeador',
        'RECURRENTE':'Recurrente','FUGA':'Fuga'
    }
    for df in [scored, snap]:
        if 'funnel_state_at_t0' in df.columns:
            df['funnel_state_at_t0'] = df['funnel_state_at_t0'].map(lambda x: funnel_map.get(x, x))

    return scored, snap

scored, snap = load_data()

# ── Sidebar: filtros globales ────────────────────────────────
st.sidebar.title("🎯 Loyalty Intelligence")
st.sidebar.markdown("**50K clientes reales**")

vista = st.sidebar.radio("Vista", [
    "📊 KPIs Ejecutivos",
    "🔄 Funnel Markov",
    "🧩 Segmentacion",
    "⚡ Incrementalidad",
    "📈 Modelo & Lift",
    "👤 Customer 360",
    "📋 Aperturas",
    "💾 Exports",
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Filtros globales")

# Filtros
tiers_avail = sorted(scored.tier.dropna().unique()) if 'tier' in scored.columns else []
tiers = st.sidebar.multiselect("Tier", tiers_avail, default=tiers_avail)

clusters_avail = sorted(scored.cluster_name.dropna().unique()) if 'cluster_name' in scored.columns else []
clusters = st.sidebar.multiselect("Cluster", clusters_avail, default=clusters_avail)

funnels_avail = sorted(scored.funnel_state_at_t0.dropna().unique()) if 'funnel_state_at_t0' in scored.columns else []
funnels = st.sidebar.multiselect("Funnel", funnels_avail, default=funnels_avail)

tipos_avail = sorted(scored.tipo_cliente.dropna().unique()) if 'tipo_cliente' in scored.columns else []
tipos = st.sidebar.multiselect("Tipo cliente", tipos_avail, default=tipos_avail)

quintiles_avail = sorted(scored.quintil_gasto.dropna().unique()) if 'quintil_gasto' in scored.columns else []
quintiles = st.sidebar.multiselect("Quintil gasto", quintiles_avail, default=quintiles_avail)

prioridades_avail = ['Alta','Media','Baja','No contactar']
prioridades = st.sidebar.multiselect("Prioridad", prioridades_avail, default=prioridades_avail)

# Aplicar filtros
def filtrar(df):
    m = pd.Series(True, index=df.index)
    if 'tier' in df.columns and tiers: m &= df.tier.isin(tiers)
    if 'cluster_name' in df.columns and clusters: m &= df.cluster_name.isin(clusters)
    if 'funnel_state_at_t0' in df.columns and funnels: m &= df.funnel_state_at_t0.isin(funnels)
    if 'tipo_cliente' in df.columns and tipos: m &= df.tipo_cliente.isin(tipos)
    if 'quintil_gasto' in df.columns and quintiles: m &= df.quintil_gasto.isin(quintiles)
    if 'prioridad' in df.columns and prioridades: m &= df.prioridad.isin(prioridades)
    return df[m]

df = filtrar(scored)
st.sidebar.markdown(f"**{len(df):,} filas** ({len(df)/len(scored)*100:.0f}%)")

# ══════════════════════════════════════════════════════════════
if vista == "📊 KPIs Ejecutivos":
    st.title("📊 KPIs Ejecutivos")

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Clientes", f"{df.cust_id.nunique():,}")
    k2.metric("Tasa Canje", f"{(df.y>0).mean()*100:.1f}%")
    k3.metric("Gasto Prom", f"${df.monetary_total.mean():,.0f}" if 'monetary_total' in df.columns else "N/A")
    k4.metric("Stock Pts", f"{df.stock_points_at_t0.mean():,.0f}" if 'stock_points_at_t0' in df.columns else "N/A")
    k5.metric("% Alta", f"{(df.prioridad=='Alta').mean()*100:.1f}%" if 'prioridad' in df.columns else "N/A")

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Por Tier")
        if 'tier' in df.columns:
            tier_df = df.groupby('tier').apply(lambda x: pd.Series({'N':len(x),'Tasa':(x.y>0).mean()*100,'y1':(x.y==1).mean()*100,'y2':(x.y==2).mean()*100})).reset_index()
            fig = px.bar(tier_df, x='tier', y=['y1','y2'], barmode='stack', labels={'value':'%','variable':'Target'})
            st.plotly_chart(fig, width='stretch')

    with c2:
        st.subheader("Por Funnel")
        if 'funnel_state_at_t0' in df.columns:
            fn_df = df.groupby('funnel_state_at_t0').apply(lambda x: pd.Series({'N':len(x),'Tasa':(x.y>0).mean()*100})).reset_index().sort_values('Tasa',ascending=False)
            fig = px.bar(fn_df, x='funnel_state_at_t0', y='Tasa', color='Tasa', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, width='stretch')

    # Evolución temporal
    st.subheader("Evolución temporal")
    snap_f = filtrar(snap) if len(snap)>0 else snap
    if 't0' in snap_f.columns and len(snap_f)>0:
        evo = snap_f.groupby('t0').apply(lambda x: (x.y>0).mean()*100).reset_index()
        evo.columns = ['t0','tasa']
        fig = px.line(evo, x='t0', y='tasa', markers=True)
        fig.update_layout(yaxis_title="Tasa canje %")
        st.plotly_chart(fig, width='stretch')

# ══════════════════════════════════════════════════════════════
elif vista == "🔄 Funnel Markov":
    st.title("🔄 Funnel Markov")

    states = ['INSCRITO','PARTICIPANTE','POSIBILIDAD_CANJE','CANJEADOR','RECURRENTE','FUGA']

    snap_f = filtrar(snap)
    if 't0' in snap_f.columns and 'funnel_state_at_t0' in snap_f.columns:
        evo = snap_f.groupby(['t0','funnel_state_at_t0']).size().reset_index(name='n')
        fig = px.area(evo, x='t0', y='n', color='funnel_state_at_t0', category_orders={'funnel_state_at_t0':states})
        fig.update_layout(yaxis_title="Clientes", legend_title="Estado")
        st.plotly_chart(fig, width='stretch')

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Distribución actual")
        fn_dist = df.funnel_state_at_t0.value_counts()
        fig = px.pie(values=fn_dist.values, names=fn_dist.index, hole=0.4)
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.subheader("Tasa canje por estado")
        fn_rate = df.groupby('funnel_state_at_t0').apply(lambda x: (x.y>0).mean()*100).reset_index()
        fn_rate.columns = ['Estado','Tasa']
        fig = px.bar(fn_rate.sort_values('Tasa',ascending=False), x='Estado', y='Tasa')
        st.plotly_chart(fig, width='stretch')

# ══════════════════════════════════════════════════════════════
elif vista == "🧩 Segmentacion":
    st.title("🧩 Segmentacion — Clusters")

    if 'cluster_name' in df.columns:
        cl_df = df.groupby('cluster_name').apply(lambda x: pd.Series({
            'N':len(x), 'Tasa canje':(x.y>0).mean()*100,
            'Propensity':x.propensity_score.mean() if 'propensity_score' in x.columns else 0,
            'Uplift':x.uplift_x.mean() if 'uplift_x' in x.columns else 0,
            'EV':x.expected_value.mean() if 'expected_value' in x.columns else 0,
            '% Alta':(x.prioridad=='Alta').mean()*100 if 'prioridad' in x.columns else 0,
        })).reset_index()
        st.dataframe(cl_df.style.format({'N':'{:,.0f}','Tasa canje':'{:.1f}%','Propensity':'{:.3f}','Uplift':'${:+,.0f}','EV':'${:+,.0f}','% Alta':'{:.1f}%'}), width='stretch')

        c1,c2 = st.columns(2)
        with c1:
            fig = px.pie(cl_df, names='cluster_name', values='N', hole=0.4, title="Distribución")
            st.plotly_chart(fig, width='stretch')
        with c2:
            if 'prioridad' in df.columns:
                cross = pd.crosstab(df.cluster_name, df.prioridad, normalize='index')*100
                fig = px.bar(cross.reset_index(), x='cluster_name', y=[c for c in ['Alta','Media','Baja','No contactar'] if c in cross.columns], barmode='stack', title="Prioridad por cluster")
                st.plotly_chart(fig, width='stretch')

# ══════════════════════════════════════════════════════════════
elif vista == "⚡ Incrementalidad":
    st.title("⚡ Incrementalidad — GitLab por Retail")

    st.markdown("**Resultados de la réplica GitLab 3 periodos (PREV/SEL/POST)**")

    gitlab_data = pd.DataFrame([
        {'Retail':'FALABELLA','Canjeadores':6640,'Potenciales':5593,'Lift gasto %':28.1},
        {'Retail':'SODIMAC','Canjeadores':6086,'Potenciales':5656,'Lift gasto %':25.0},
        {'Retail':'TOTTUS','Canjeadores':4907,'Potenciales':3938,'Lift gasto %':37.0},
        {'Retail':'FCOM','Canjeadores':4820,'Potenciales':2636,'Lift gasto %':19.3},
        {'Retail':'TOTAL','Canjeadores':8904,'Potenciales':10245,'Lift gasto %':35.3},
    ])
    st.dataframe(gitlab_data.style.format({'Canjeadores':'{:,}','Potenciales':'{:,}','Lift gasto %':'{:+.1f}%'}), width='stretch')

    fig = px.bar(gitlab_data[gitlab_data.Retail!='TOTAL'], x='Retail', y='Lift gasto %', color='Lift gasto %', color_continuous_scale='RdYlGn', title="Lift gasto % por retail")
    st.plotly_chart(fig, width='stretch')

    if 'uplift_x' in df.columns:
        st.subheader("Distribución de Uplift")
        fig2 = px.histogram(df, x='uplift_x', nbins=100, color='prioridad' if 'prioridad' in df.columns else None,
            color_discrete_map={'Alta':'green','Media':'orange','Baja':'gray','No contactar':'red'})
        fig2.update_layout(xaxis_title="Uplift (CLP)")
        st.plotly_chart(fig2, width='stretch')

# ══════════════════════════════════════════════════════════════
elif vista == "📈 Modelo & Lift":
    st.title("📈 Modelo & Lift")

    if 'propensity_score' in df.columns and 'y' in df.columns:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            y_bin = (df.y>0).astype(int)
            prop = df.propensity_score.fillna(0)
            auc = roc_auc_score(y_bin, prop)
            st.metric("AUC", f"{auc:.4f}")

            fpr,tpr,_ = roc_curve(y_bin, prop)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f'AUC={auc:.4f}'))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(dash='dash'),name='Random'))
            fig.update_layout(xaxis_title='FPR',yaxis_title='TPR',title='ROC Curve')
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.warning(f"No se pudo calcular AUC: {e}")

        # Lift por decil
        st.subheader("Lift por decil")
        base = y_bin.mean()
        df_tmp = df.copy()
        df_tmp['dec'] = pd.qcut(df_tmp.propensity_score.rank(method='first'),10,labels=range(1,11))
        lift_data = []
        cum=0; total=y_bin.sum()
        for d in range(10,0,-1):
            s=df_tmp[df_tmp.dec==d]; r=(s.y>0).sum(); cum+=r
            lift_data.append({'Decil':d,'Rate':r/len(s)*100,'Lift':r/len(s)/base,'Cum':cum/total*100})
        lift_df = pd.DataFrame(lift_data)

        fig2 = make_subplots(specs=[[{"secondary_y":True}]])
        fig2.add_trace(go.Bar(x=lift_df.Decil,y=lift_df.Lift,name='Lift',marker_color='steelblue'),secondary_y=False)
        fig2.add_trace(go.Scatter(x=lift_df.Decil,y=lift_df.Cum,name='Cum Capture %',mode='lines+markers',marker_color='red'),secondary_y=True)
        fig2.update_layout(xaxis_title='Decil (10=top)',xaxis=dict(dtick=1))
        fig2.update_yaxes(title_text='Lift',secondary_y=False)
        fig2.update_yaxes(title_text='Cum Capture %',secondary_y=True)
        st.plotly_chart(fig2, width='stretch')

        st.dataframe(lift_df.style.format({'Rate':'{:.1f}%','Lift':'{:.2f}x','Cum':'{:.1f}%'}))

# ══════════════════════════════════════════════════════════════
elif vista == "👤 Customer 360":
    st.title("👤 Customer 360")

    cust_id = st.text_input("cust_id", value=scored.cust_id.iloc[0])
    cust = scored[scored.cust_id==cust_id]

    if len(cust)==0:
        st.error(f"Cliente {cust_id} no encontrado")
    else:
        r = cust.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Tier", str(r.get('tier','?')))
        c2.metric("Funnel", str(r.get('funnel_state_at_t0','?')))
        c3.metric("Cluster", str(r.get('cluster_name','?')))
        c4.metric("Prioridad", str(r.get('prioridad','?')))

        c5,c6,c7,c8 = st.columns(4)
        c5.metric("Propensity", f"{r.propensity_score:.3f}" if 'propensity_score' in r.index else "?")
        c6.metric("EV", f"${r.expected_value:+,.0f}" if 'expected_value' in r.index else "?")
        c7.metric("Gasto 12m", f"${r.monetary_total:,.0f}" if 'monetary_total' in r.index else "?")
        c8.metric("Stock pts", f"{r.stock_points_at_t0:,.0f}" if 'stock_points_at_t0' in r.index else "?")

        st.subheader("Recomendación")
        rec = {k:str(r.get(k,'N/A')) for k in ['objetivo','accion','canal','timing']}
        st.json(rec)

        # Historia
        hist = snap[snap.cust_id==cust_id].sort_values('t0') if 't0' in snap.columns else pd.DataFrame()
        if len(hist)>0 and 'monetary_total' in hist.columns:
            st.subheader("Evolución")
            fig = make_subplots(rows=2,cols=1,subplot_titles=["Gasto 12m","Stock puntos"])
            fig.add_trace(go.Scatter(x=hist.t0,y=hist.monetary_total,mode='lines+markers'),row=1,col=1)
            if 'stock_points_at_t0' in hist.columns:
                fig.add_trace(go.Scatter(x=hist.t0,y=hist.stock_points_at_t0,mode='lines+markers'),row=2,col=1)
            fig.update_layout(height=500,showlegend=False)
            st.plotly_chart(fig, width='stretch')

# ══════════════════════════════════════════════════════════════
elif vista == "📋 Aperturas":
    st.title("📋 Aperturas detalladas")

    # Por quintil
    if 'quintil_gasto' in df.columns:
        st.subheader("Por quintil de gasto")
        q_df = df.groupby('quintil_gasto').apply(lambda x: pd.Series({
            'N':len(x),'Tasa':(x.y>0).mean()*100,
            'Prop':x.propensity_score.mean() if 'propensity_score' in x.columns else 0,
            '%Alta':(x.prioridad=='Alta').mean()*100 if 'prioridad' in x.columns else 0,
        })).reset_index()
        st.dataframe(q_df.style.format({'N':'{:,.0f}','Tasa':'{:.1f}%','Prop':'{:.3f}','%Alta':'{:.1f}%'}), width='stretch')

    # Cross: quintil × tier
    if 'quintil_gasto' in df.columns and 'tier' in df.columns:
        st.subheader("Tasa canje: Quintil × Tier")
        cross = df.groupby(['quintil_gasto','tier']).apply(lambda x: (x.y>0).mean()*100).unstack(fill_value=0)
        st.dataframe(cross.style.format('{:.1f}%'), width='stretch')

    # Cross: cluster × funnel
    if 'cluster_name' in df.columns and 'funnel_state_at_t0' in df.columns:
        st.subheader("N clientes: Cluster × Funnel")
        cross2 = pd.crosstab(df.cluster_name, df.funnel_state_at_t0)
        st.dataframe(cross2.style.format('{:,}'), width='stretch')

    # Por tipo cliente
    if 'tipo_cliente' in df.columns:
        st.subheader("Por tipo cliente")
        tc_df = df.groupby('tipo_cliente').apply(lambda x: pd.Series({
            'N':len(x),'Tasa':(x.y>0).mean()*100,
            'Prop':x.propensity_score.mean() if 'propensity_score' in x.columns else 0,
            'Gasto':x.monetary_total.mean() if 'monetary_total' in x.columns else 0,
        })).reset_index()
        st.dataframe(tc_df.style.format({'N':'{:,.0f}','Tasa':'{:.1f}%','Prop':'{:.3f}','Gasto':'${:,.0f}'}), width='stretch')

    # Prioridad × Quintil
    if 'prioridad' in df.columns and 'quintil_gasto' in df.columns:
        st.subheader("Prioridad × Quintil")
        cross3 = pd.crosstab(df.quintil_gasto, df.prioridad, normalize='index')*100
        st.dataframe(cross3.style.format('{:.1f}%'), width='stretch')

# ══════════════════════════════════════════════════════════════
elif vista == "💾 Exports":
    st.title("💾 Exports")

    for group_col, group_name in [('prioridad','Prioridad'),('cluster_name','Cluster')]:
        if group_col not in df.columns: continue
        st.subheader(f"Por {group_name}")
        for val in sorted(df[group_col].dropna().unique()):
            sub = df[df[group_col]==val]
            cols = [c for c in ['cust_id','propensity_score','uplift_x','expected_value','prioridad','cluster_name','funnel_state_at_t0','tier','canal','timing'] if c in sub.columns]
            st.markdown(f"**{val}** ({len(sub):,} clientes)")
            csv = sub[cols].to_csv(index=False)
            st.download_button(f"Descargar {val}", csv, f"{group_name.lower()}_{str(val).lower().replace(' ','_')}.csv", "text/csv")

    st.subheader("Scoring completo")
    cols_all = [c for c in ['cust_id','t0','tier','funnel_state_at_t0','propensity_score','uplift_x','expected_value','prioridad','cluster_name','canal','timing','y'] if c in df.columns]
    csv_all = df[cols_all].to_csv(index=False)
    st.download_button("Descargar scoring completo", csv_all, "scoring_completo.csv", "text/csv")
