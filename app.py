"""
Loyalty Intelligence System — Dashboard
50K clientes reales CMR Puntos | 8 vistas | Filtros globales
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Loyalty Intelligence", page_icon="🎯", layout="wide")

@st.cache_data(ttl=3600, show_spinner="Cargando datos...")
def load_data():
    scored = pd.read_parquet('data_scored.parquet')
    temporal = pd.read_csv('data_temporal.csv')
    funnel_evo = pd.read_csv('data_funnel_evo.csv')

    # Fix types
    for c in scored.columns:
        if scored[c].dtype == 'object' and c not in ['cust_id','tier','gender','city','dominant_retailer','funnel_state_at_t0','status','cluster_name','prioridad','canal','timing','objetivo','accion']:
            try: scored[c] = pd.to_numeric(scored[c], errors='coerce').fillna(0)
            except: pass

    # Tipo cliente
    if 'stock_points_at_t0' in scored.columns and 'y' in scored.columns:
        scored['tipo_cliente'] = np.where(scored.y>=1, 'CANJEADOR',
            np.where(scored.stock_points_at_t0>=1000, 'POTENCIAL', 'ACUMULADOR'))

    # Quintiles
    if 'quintil_gasto' not in scored.columns and 'monetary_total' in scored.columns:
        scored['quintil_gasto'] = pd.qcut(scored.monetary_total.rank(method='first'), 5, labels=['Q1','Q2','Q3','Q4','Q5'])

    # Renombrar tiers
    tier_map = {'NORMAL':'Entrada','Normal':'Entrada','FAN':'Fan','PREMIUM':'Premium','ELITE':'Elite'}
    scored['tier'] = scored['tier'].map(lambda x: tier_map.get(x, x))

    # Renombrar funnel
    fn_map = {'INSCRITO':'Inscrito','PARTICIPANTE':'Participante','POSIBILIDAD_CANJE':'Posibilidad Canje','CANJEADOR':'Canjeador','RECURRENTE':'Recurrente','FUGA':'Fuga'}
    scored['funnel_state_at_t0'] = scored['funnel_state_at_t0'].map(lambda x: fn_map.get(x, x))
    if 'funnel_state_at_t0' in funnel_evo.columns:
        funnel_evo['funnel_state_at_t0'] = funnel_evo['funnel_state_at_t0'].map(lambda x: fn_map.get(x, x))

    return scored, temporal, funnel_evo

scored, temporal, funnel_evo = load_data()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🎯 Loyalty Intelligence")
st.sidebar.markdown("**50K clientes reales**")

vista = st.sidebar.radio("Vista", [
    "📊 KPIs Ejecutivos", "🔄 Funnel Markov", "🧩 Segmentacion",
    "⚡ Incrementalidad", "📈 Modelo & Lift", "👤 Customer 360",
    "📋 Aperturas", "💾 Exports",
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Filtros")

tiers = st.sidebar.multiselect("Tier", sorted(scored.tier.dropna().unique()), default=sorted(scored.tier.dropna().unique()))
clusters = st.sidebar.multiselect("Cluster", sorted(scored.cluster_name.dropna().unique()), default=sorted(scored.cluster_name.dropna().unique())) if 'cluster_name' in scored.columns else []
funnels = st.sidebar.multiselect("Funnel", sorted(scored.funnel_state_at_t0.dropna().unique()), default=sorted(scored.funnel_state_at_t0.dropna().unique()))
tipos = st.sidebar.multiselect("Tipo cliente", sorted(scored.tipo_cliente.dropna().unique()), default=sorted(scored.tipo_cliente.dropna().unique())) if 'tipo_cliente' in scored.columns else []
quintiles = st.sidebar.multiselect("Quintil", sorted(scored.quintil_gasto.dropna().unique()), default=sorted(scored.quintil_gasto.dropna().unique())) if 'quintil_gasto' in scored.columns else []
prioridades = st.sidebar.multiselect("Prioridad", ['Alta','Media','Baja','No contactar'], default=['Alta','Media','Baja','No contactar'])

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
st.sidebar.metric("Filas filtradas", f"{len(df):,}")

# ══════════════════════════════════════════════════════════════
if vista == "📊 KPIs Ejecutivos":
    st.title("📊 KPIs Ejecutivos")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Clientes", f"{df.cust_id.nunique():,}")
    k2.metric("Tasa Canje", f"{(df.y>0).mean()*100:.1f}%")
    k3.metric("Gasto Prom", f"${df.monetary_total.mean():,.0f}" if 'monetary_total' in df.columns else "-")
    k4.metric("Stock Pts", f"{df.stock_points_at_t0.mean():,.0f}" if 'stock_points_at_t0' in df.columns else "-")
    k5.metric("% Alta", f"{(df.prioridad=='Alta').mean()*100:.1f}%" if 'prioridad' in df.columns else "-")

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Por Tier")
        t_df = df.groupby('tier').apply(lambda x: pd.Series({'Tasa':(x.y>0).mean()*100,'y1':(x.y==1).mean()*100,'y2':(x.y==2).mean()*100})).reset_index()
        fig = px.bar(t_df, x='tier', y=['y1','y2'], barmode='stack', labels={'value':'%'})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Por Funnel")
        f_df = df.groupby('funnel_state_at_t0').apply(lambda x: (x.y>0).mean()*100).reset_index(name='Tasa').sort_values('Tasa',ascending=False)
        fig = px.bar(f_df, x='funnel_state_at_t0', y='Tasa', color='Tasa', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evolucion temporal")
    fig = px.line(temporal, x='t0', y='tasa_canje', markers=True)
    fig.update_layout(yaxis_title="Tasa canje %")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif vista == "🔄 Funnel Markov":
    st.title("🔄 Funnel Markov")
    states = ['Inscrito','Participante','Posibilidad Canje','Canjeador','Recurrente','Fuga']
    fig = px.area(funnel_evo, x='t0', y='n', color='funnel_state_at_t0', category_orders={'funnel_state_at_t0':states})
    fig.update_layout(yaxis_title="Clientes", legend_title="Estado")
    st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Distribucion actual")
        fig = px.pie(df, names='funnel_state_at_t0', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Tasa canje por estado")
        r = df.groupby('funnel_state_at_t0').apply(lambda x: (x.y>0).mean()*100).reset_index(name='Tasa').sort_values('Tasa',ascending=False)
        fig = px.bar(r, x='funnel_state_at_t0', y='Tasa')
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif vista == "🧩 Segmentacion":
    st.title("🧩 Segmentacion")
    if 'cluster_name' in df.columns:
        cl = df.groupby('cluster_name').apply(lambda x: pd.Series({
            'N':len(x),'Tasa':(x.y>0).mean()*100,
            'Prop':x.propensity_score.mean() if 'propensity_score' in x.columns else 0,
            'Uplift':x.uplift_x.mean() if 'uplift_x' in x.columns else 0,
            'EV':x.expected_value.mean() if 'expected_value' in x.columns else 0,
            '%Alta':(x.prioridad=='Alta').mean()*100 if 'prioridad' in x.columns else 0,
        })).reset_index()
        st.dataframe(cl.style.format({'N':'{:,.0f}','Tasa':'{:.1f}%','Prop':'{:.3f}','Uplift':'${:+,.0f}','EV':'${:+,.0f}','%Alta':'{:.1f}%'}))

        c1,c2 = st.columns(2)
        with c1:
            fig = px.pie(cl, names='cluster_name', values='N', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            cross = pd.crosstab(df.cluster_name, df.prioridad, normalize='index')*100
            fig = px.bar(cross.reset_index(), x='cluster_name', y=[c for c in ['Alta','Media','Baja','No contactar'] if c in cross.columns], barmode='stack')
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif vista == "⚡ Incrementalidad":
    st.title("⚡ Incrementalidad GitLab por Retail")
    gl = pd.DataFrame([
        {'Retail':'FALABELLA','Cj':6640,'Pt':5593,'Lift':28.1},
        {'Retail':'SODIMAC','Cj':6086,'Pt':5656,'Lift':25.0},
        {'Retail':'TOTTUS','Cj':4907,'Pt':3938,'Lift':37.0},
        {'Retail':'FCOM','Cj':4820,'Pt':2636,'Lift':19.3},
        {'Retail':'TOTAL','Cj':8904,'Pt':10245,'Lift':35.3},
    ])
    c1,c2,c3 = st.columns(3)
    c1.metric("Lift global", "+35.3%")
    c2.metric("Canjeadores", "8,904")
    c3.metric("Potenciales", "10,245")

    fig = px.bar(gl[gl.Retail!='TOTAL'], x='Retail', y='Lift', color='Lift', color_continuous_scale='RdYlGn', title="Lift gasto % por retail")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(gl.style.format({'Cj':'{:,}','Pt':'{:,}','Lift':'{:+.1f}%'}))

    if 'uplift_x' in df.columns:
        st.subheader("Distribucion Uplift")
        fig = px.histogram(df, x='uplift_x', nbins=80, color='prioridad' if 'prioridad' in df.columns else None,
            color_discrete_map={'Alta':'green','Media':'orange','Baja':'gray','No contactar':'red'})
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif vista == "📈 Modelo & Lift":
    st.title("📈 Modelo & Lift")
    if 'propensity_score' in df.columns and 'y' in df.columns:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            yb=(df.y>0).astype(int); pp=df.propensity_score.fillna(0)
            auc=roc_auc_score(yb,pp)
            st.metric("AUC", f"{auc:.4f}")
            fpr,tpr,_=roc_curve(yb,pp)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f'AUC={auc:.4f}'))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(dash='dash'),name='Random'))
            fig.update_layout(xaxis_title='FPR',yaxis_title='TPR',title='ROC Curve')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Error AUC: {e}")

        st.subheader("Lift por decil")
        base=yb.mean()
        dft=df.copy(); dft['dec']=pd.qcut(dft.propensity_score.rank(method='first'),10,labels=range(1,11))
        ld=[]; cum=0; tot=yb.sum()
        for d in range(10,0,-1):
            s=dft[dft.dec==d]; r=(s.y>0).sum(); cum+=r
            ld.append({'Decil':d,'Rate':r/len(s)*100,'Lift':r/len(s)/base,'Cum':cum/tot*100})
        ldf=pd.DataFrame(ld)
        fig2=make_subplots(specs=[[{"secondary_y":True}]])
        fig2.add_trace(go.Bar(x=ldf.Decil,y=ldf.Lift,name='Lift',marker_color='steelblue'),secondary_y=False)
        fig2.add_trace(go.Scatter(x=ldf.Decil,y=ldf.Cum,name='Cum %',mode='lines+markers',marker_color='red'),secondary_y=True)
        fig2.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(ldf.style.format({'Rate':'{:.1f}%','Lift':'{:.2f}x','Cum':'{:.1f}%'}))

# ══════════════════════════════════════════════════════════════
elif vista == "👤 Customer 360":
    st.title("👤 Customer 360")
    cid=st.text_input("cust_id", value=scored.cust_id.iloc[0])
    c=scored[scored.cust_id==cid]
    if len(c)==0: st.error("No encontrado")
    else:
        r=c.iloc[0]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Tier",str(r.get('tier','?'))); c2.metric("Funnel",str(r.get('funnel_state_at_t0','?')))
        c3.metric("Cluster",str(r.get('cluster_name','?'))); c4.metric("Prioridad",str(r.get('prioridad','?')))
        c5,c6,c7,c8=st.columns(4)
        c5.metric("Propensity",f"{r.propensity_score:.3f}" if 'propensity_score' in r.index else "?")
        c6.metric("EV",f"${r.expected_value:+,.0f}" if 'expected_value' in r.index else "?")
        c7.metric("Gasto",f"${r.monetary_total:,.0f}" if 'monetary_total' in r.index else "?")
        c8.metric("Puntos",f"{r.stock_points_at_t0:,.0f}" if 'stock_points_at_t0' in r.index else "?")
        st.json({k:str(r.get(k,'N/A')) for k in ['objetivo','accion','canal','timing']})

# ══════════════════════════════════════════════════════════════
elif vista == "📋 Aperturas":
    st.title("📋 Aperturas")

    if 'quintil_gasto' in df.columns:
        st.subheader("Por quintil de gasto")
        q=df.groupby('quintil_gasto').apply(lambda x: pd.Series({'N':len(x),'Tasa':(x.y>0).mean()*100,'Prop':x.propensity_score.mean() if 'propensity_score' in x.columns else 0,'%Alta':(x.prioridad=='Alta').mean()*100 if 'prioridad' in x.columns else 0})).reset_index()
        st.dataframe(q.style.format({'N':'{:,.0f}','Tasa':'{:.1f}%','Prop':'{:.3f}','%Alta':'{:.1f}%'}))

    if 'quintil_gasto' in df.columns and 'tier' in df.columns:
        st.subheader("Tasa canje: Quintil x Tier")
        ct=df.groupby(['quintil_gasto','tier']).apply(lambda x: (x.y>0).mean()*100).unstack(fill_value=0)
        st.dataframe(ct.style.format('{:.1f}%'))

    if 'cluster_name' in df.columns and 'funnel_state_at_t0' in df.columns:
        st.subheader("Cluster x Funnel")
        st.dataframe(pd.crosstab(df.cluster_name, df.funnel_state_at_t0).style.format('{:,}'))

    if 'tipo_cliente' in df.columns:
        st.subheader("Por tipo cliente")
        tc=df.groupby('tipo_cliente').apply(lambda x: pd.Series({'N':len(x),'Tasa':(x.y>0).mean()*100,'Gasto':x.monetary_total.mean() if 'monetary_total' in x.columns else 0})).reset_index()
        st.dataframe(tc.style.format({'N':'{:,.0f}','Tasa':'{:.1f}%','Gasto':'${:,.0f}'}))

    if 'prioridad' in df.columns and 'quintil_gasto' in df.columns:
        st.subheader("Prioridad x Quintil")
        st.dataframe((pd.crosstab(df.quintil_gasto, df.prioridad, normalize='index')*100).style.format('{:.1f}%'))

# ══════════════════════════════════════════════════════════════
elif vista == "💾 Exports":
    st.title("💾 Exports")
    for gc,gn in [('prioridad','Prioridad'),('cluster_name','Cluster')]:
        if gc not in df.columns: continue
        st.subheader(f"Por {gn}")
        for v in sorted(df[gc].dropna().unique()):
            s=df[df[gc]==v]
            cols=[c for c in ['cust_id','propensity_score','uplift_x','expected_value','prioridad','cluster_name','funnel_state_at_t0','tier','canal','timing'] if c in s.columns]
            csv=s[cols].to_csv(index=False)
            st.download_button(f"{v} ({len(s):,})", csv, f"{gn.lower()}_{str(v).lower().replace(' ','_')}.csv", "text/csv")
