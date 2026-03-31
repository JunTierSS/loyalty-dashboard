"""
🎯 Loyalty Intelligence System — CMR Puntos
Dashboard de Inteligencia para toma de decisiones
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Loyalty Intelligence", page_icon="🎯", layout="wide")

# ── Data ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Cargando datos...")
def load():
    df = pd.read_csv('data_scored.csv.gz', compression='gzip')
    temporal = pd.read_csv('data_temporal.csv')
    funnel_evo = pd.read_csv('data_funnel_evo.csv')
    metrics = json.load(open('data_metrics.json'))

    # Ensure numeric
    for c in df.columns:
        if c not in ['cust_id','tier','gender','city','dominant_retailer','funnel_state_at_t0','status',
                     'cluster_name','prioridad','canal','timing','objetivo','accion','tipo_cliente',
                     'quintil_label','target_label','t0']:
            try: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            except: pass

    # Quintil label
    if 'quintil_label' not in df.columns and 'quintil_gasto' in df.columns:
        df['quintil_label'] = df['quintil_gasto'].map({1:'Q1',2:'Q2',3:'Q3',4:'Q4',5:'Q5'})

    # Tipo cliente
    if 'tipo_cliente' not in df.columns:
        df['tipo_cliente'] = np.where(df.y>=1, 'Canjeador', np.where(df.stock_points_at_t0>=1000, 'Potencial', 'Acumulador'))

    # Target label
    if 'target_label' not in df.columns:
        df['target_label'] = df['y'].map({0:'No canjea',1:'Activacion',2:'Recurrencia'})

    # CLV
    if 'clv_estimado' not in df.columns:
        df['clv_estimado'] = df['revenue_post_12m'] * 1.5

    return df, temporal, funnel_evo, metrics

df_all, temporal, funnel_evo, metrics = load()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Falabella_logo.svg/200px-Falabella_logo.svg.png", width=120)
st.sidebar.title("Loyalty Intelligence")
st.sidebar.caption(f"{metrics['n_clientes']:,} clientes reales")

vista = st.sidebar.radio("", [
    "🏠 Resumen Ejecutivo",
    "💰 Valor del Cliente (CLV)",
    "🔄 Funnel de Canje",
    "🧩 Segmentos & Clusters",
    "📈 Performance del Modelo",
    "⚡ Incrementalidad",
    "🎯 Motor de Recomendaciones",
    "👤 Ficha Cliente",
    "📋 Aperturas & Cross-tabs",
    "💾 Exportar Listas",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros**")

# Filtros
tiers_all = sorted([x for x in df_all.tier.dropna().unique() if x != 'UNKNOWN'])
tiers = st.sidebar.multiselect("Categoria", tiers_all, default=tiers_all, key='tier')

if 'cluster_name' in df_all.columns:
    cl_all = sorted([x for x in df_all.cluster_name.dropna().unique() if x not in ['',None]])
    clusters = st.sidebar.multiselect("Cluster", cl_all, default=cl_all, key='cl')
else: clusters = []

fn_all = sorted([x for x in df_all.funnel_state_at_t0.dropna().unique() if x != 'UNKNOWN'])
funnels = st.sidebar.multiselect("Estado Funnel", fn_all, default=fn_all, key='fn')

qt_all = sorted([x for x in df_all.quintil_label.dropna().unique() if x not in ['',None]]) if 'quintil_label' in df_all.columns else []
quintiles = st.sidebar.multiselect("Quintil Gasto", qt_all, default=qt_all, key='qt')

pri_all = ['Alta','Media','Baja','No contactar']
prioridades = st.sidebar.multiselect("Prioridad", pri_all, default=pri_all, key='pri')

def filt(d):
    m = pd.Series(True, index=d.index)
    if tiers: m &= d.tier.isin(tiers)
    if clusters and 'cluster_name' in d.columns: m &= d.cluster_name.isin(clusters)
    if funnels: m &= d.funnel_state_at_t0.isin(funnels)
    if quintiles and 'quintil_label' in d.columns: m &= d.quintil_label.isin(quintiles)
    if prioridades and 'prioridad' in d.columns: m &= d.prioridad.isin(prioridades)
    return d[m]

df = filt(df_all)
st.sidebar.caption(f"📊 {len(df):,} registros filtrados")

# ══════════════════════════════════════════════════════════════
# VISTA 1: RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════
if vista == "🏠 Resumen Ejecutivo":
    st.title("🏠 Resumen Ejecutivo")
    st.markdown("**Vista general del programa de lealtad y oportunidades de negocio**")

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Clientes", f"{df.cust_id.nunique():,}")
    k2.metric("Tasa Canje", f"{(df.y>0).mean()*100:.1f}%")
    k3.metric("AUC Modelo", f"{metrics['auc']:.2f}")
    k4.metric("Prioridad Alta", f"{(df.prioridad=='Alta').mean()*100:.1f}%")
    avg_gasto = df.monetary_total.mean() if 'monetary_total' in df.columns else 0
    k5.metric("Gasto Prom 12m", f"${avg_gasto:,.0f}")
    k6.metric("Uplift %>0", f"{(df.uplift_x>0).mean()*100:.0f}%" if 'uplift_x' in df.columns else "-")

    st.markdown("---")

    # Insight boxes
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("💡 Oportunidad de Activacion")
        pos = df[df.funnel_state_at_t0=='Posibilidad Canje']
        avg_rev_canj = df[df.y>=1].revenue_post_12m.mean() if (df.y>=1).any() else 0
        pot_rev = len(pos) * avg_rev_canj * 0.15
        st.metric("Clientes con puntos listos para canjear", f"{len(pos):,}")
        st.metric("Revenue potencial (15% conversion)", f"${pot_rev:,.0f}")
        st.info(f"Si activamos al 15% de los {len(pos):,} clientes en **Posibilidad de Canje**, generamos **${pot_rev:,.0f}** en revenue incremental.")

    with c2:
        st.subheader("⚠️ Riesgo de Fuga")
        fuga = df[df.funnel_state_at_t0=='Fuga']
        if 'exp_points_current_at_t0' in df.columns:
            pts_fuga = fuga.exp_points_current_at_t0.sum()
        else: pts_fuga = 0
        st.metric("Clientes en Fuga", f"{len(fuga):,}")
        st.metric("Puntos en riesgo de vencer", f"{pts_fuga:,.0f}")
        st.warning(f"**{len(fuga):,}** clientes dejaron de canjear. Tienen **{pts_fuga:,.0f}** puntos acumulados en riesgo.")

    # Por categoria
    st.subheader("Tasa de canje por categoria de cliente")
    tier_df = df.groupby('tier').apply(lambda x: pd.Series({
        'Clientes':x.cust_id.nunique(), 'Tasa canje':(x.y>0).mean()*100,
        'Activacion (nuevos)':(x.y==1).mean()*100, 'Recurrencia':(x.y==2).mean()*100,
        'Gasto promedio':x.monetary_total.mean()
    })).reset_index()
    fig = px.bar(tier_df, x='tier', y=['Activacion (nuevos)','Recurrencia'], barmode='stack',
        title="Tasa de canje: nuevos canjeadores vs recurrentes", labels={'value':'%','tier':'Categoria'},
        color_discrete_sequence=['#636EFA','#EF553B'])
    fig.update_layout(yaxis_title="% de clientes que canjean", legend_title="Tipo")
    st.plotly_chart(fig, use_container_width=True)

    # Evolución
    st.subheader("Evolucion temporal de la tasa de canje")
    fig2 = px.line(temporal, x='t0', y='tasa_canje', markers=True, title="Tasa de canje mensual")
    fig2.update_layout(yaxis_title="% clientes que canjean", xaxis_title="Mes de corte (t0)")
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# VISTA 2: CLV
# ══════════════════════════════════════════════════════════════
elif vista == "💰 Valor del Cliente (CLV)":
    st.title("💰 Valor del Cliente (CLV)")
    st.markdown("**Cuanto vale cada cliente segun su comportamiento de canje**")

    c1,c2,c3 = st.columns(3)
    clv_canj = df[df.y>=1].clv_estimado.mean() if (df.y>=1).any() else 0
    clv_no = df[df.y==0].clv_estimado.mean() if (df.y==0).any() else 0
    ratio = clv_canj/clv_no if clv_no>0 else 0
    c1.metric("CLV Canjeador", f"${clv_canj:,.0f}")
    c2.metric("CLV No canjeador", f"${clv_no:,.0f}")
    c3.metric("Diferencia", f"{ratio:.1f}x")
    st.success(f"Un cliente que canjea vale **{ratio:.1f}x mas** que uno que no. Activar canjes = multiplicar el valor del cliente.")

    # CLV por tier
    st.subheader("CLV por categoria y comportamiento")
    clv_tier = df.groupby(['tier','target_label']).clv_estimado.mean().reset_index()
    fig = px.bar(clv_tier, x='tier', y='clv_estimado', color='target_label', barmode='group',
        title="CLV promedio por categoria", labels={'clv_estimado':'CLV estimado ($)','tier':'Categoria'},
        color_discrete_map={'No canjea':'#BDBDBD','Activacion':'#42A5F5','Recurrencia':'#EF5350'})
    fig.update_layout(yaxis_title="CLV promedio ($)", legend_title="Comportamiento")
    st.plotly_chart(fig, use_container_width=True)

    # CLV por cluster
    if 'cluster_name' in df.columns:
        st.subheader("CLV por segmento")
        clv_cl = df.groupby('cluster_name').apply(lambda x: pd.Series({
            'CLV promedio':x.clv_estimado.mean(), 'N':len(x), 'Tasa canje':(x.y>0).mean()*100
        })).reset_index().sort_values('CLV promedio',ascending=False)
        fig2 = px.bar(clv_cl, x='cluster_name', y='CLV promedio', color='Tasa canje',
            color_continuous_scale='RdYlGn', title="CLV por segmento de cliente")
        fig2.update_layout(yaxis_title="CLV promedio ($)", xaxis_title="Segmento")
        st.plotly_chart(fig2, use_container_width=True)

    # CLV por quintil
    if 'quintil_label' in df.columns:
        st.subheader("CLV por quintil de gasto")
        clv_q = df.groupby('quintil_label').apply(lambda x: pd.Series({
            'CLV':x.clv_estimado.mean(),'Tasa':(x.y>0).mean()*100,'N':len(x)
        })).reset_index()
        fig3 = px.bar(clv_q, x='quintil_label', y='CLV', text=clv_q['Tasa'].apply(lambda x: f'{x:.0f}%'),
            title="CLV por quintil (% = tasa de canje)")
        fig3.update_layout(yaxis_title="CLV ($)", xaxis_title="Quintil de gasto")
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# VISTA 3: FUNNEL
# ══════════════════════════════════════════════════════════════
elif vista == "🔄 Funnel de Canje":
    st.title("🔄 Funnel de Canje")
    st.markdown("""
    **El viaje del cliente desde la inscripcion hasta el canje recurrente.**
    Cada cliente pasa por estos estados. El objetivo es moverlos hacia la derecha.
    """)

    states = ['Inscrito','Participante','Posibilidad Canje','Canjeador','Recurrente','Fuga']

    # Evolución
    fig = px.area(funnel_evo, x='t0', y='n', color='funnel_state_at_t0',
        category_orders={'funnel_state_at_t0':states},
        title="Evolucion del funnel en el tiempo",
        color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(yaxis_title="Numero de clientes", xaxis_title="Mes", legend_title="Estado")
    st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Distribucion actual")
        fn_dist = df.funnel_state_at_t0.value_counts()
        fig2 = px.pie(values=fn_dist.values, names=fn_dist.index, hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("Tasa de canje por estado")
        fn_rate = df.groupby('funnel_state_at_t0').apply(lambda x: pd.Series({
            'Tasa canje':(x.y>0).mean()*100, 'N':len(x)
        })).reset_index().sort_values('Tasa canje',ascending=False)
        fig3 = px.bar(fn_rate, x='funnel_state_at_t0', y='Tasa canje',
            color='Tasa canje', color_continuous_scale='RdYlGn',
            title="Probabilidad de canje segun estado")
        fig3.update_layout(yaxis_title="% que canjea", xaxis_title="Estado del funnel")
        st.plotly_chart(fig3, use_container_width=True)

    # Bottleneck
    st.subheader("🔍 Donde se estancan los clientes?")
    fn_counts = df.funnel_state_at_t0.value_counts()
    biggest = fn_counts.idxmax()
    st.info(f"El mayor cuello de botella es **{biggest}** con **{fn_counts[biggest]:,}** clientes ({fn_counts[biggest]/len(df)*100:.0f}%). Esto es donde las campañas deben enfocarse.")

# ══════════════════════════════════════════════════════════════
# VISTA 4: SEGMENTOS
# ══════════════════════════════════════════════════════════════
elif vista == "🧩 Segmentos & Clusters":
    st.title("🧩 Segmentos de Clientes")
    st.markdown("**5 arquetipos de comportamiento identificados por ML (K-Means)**")

    if 'cluster_name' in df.columns:
        cl_df = df.groupby('cluster_name').apply(lambda x: pd.Series({
            'Clientes':len(x), '% del total':len(x)/len(df)*100,
            'Tasa canje':(x.y>0).mean()*100,
            'Propensity':x.propensity_score.mean() if 'propensity_score' in x.columns else 0,
            'CLV promedio':x.clv_estimado.mean(),
            '% Prioridad Alta':(x.prioridad=='Alta').mean()*100 if 'prioridad' in x.columns else 0,
            'Gasto promedio':x.monetary_total.mean() if 'monetary_total' in x.columns else 0,
        })).reset_index()

        st.dataframe(cl_df.style.format({
            'Clientes':'{:,.0f}','% del total':'{:.1f}%','Tasa canje':'{:.1f}%',
            'Propensity':'{:.3f}','CLV promedio':'${:,.0f}','% Prioridad Alta':'{:.1f}%',
            'Gasto promedio':'${:,.0f}'
        }), use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            fig = px.pie(cl_df, names='cluster_name', values='Clientes', hole=0.4,
                title="Distribucion de segmentos")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'prioridad' in df.columns:
                cross = pd.crosstab(df.cluster_name, df.prioridad, normalize='index')*100
                cols_order = [c for c in ['Alta','Media','Baja','No contactar'] if c in cross.columns]
                fig2 = px.bar(cross[cols_order].reset_index(), x='cluster_name', y=cols_order,
                    barmode='stack', title="Prioridad de contacto por segmento",
                    color_discrete_map={'Alta':'#4CAF50','Media':'#FFC107','Baja':'#9E9E9E','No contactar':'#F44336'})
                fig2.update_layout(yaxis_title="%", xaxis_title="Segmento", legend_title="Prioridad")
                st.plotly_chart(fig2, use_container_width=True)

        # Acciones por cluster
        st.subheader("🎯 Accion recomendada por segmento")
        acciones = {
            'Heavy Users': '🏆 Experiencia premium, upgrade de tier, acceso anticipado',
            'Exploradores': '🔍 Ofertas multi-retailer, educar sobre beneficios del canje',
            'Cazadores de Canje': '🎁 Descuento personalizado, 20% menos puntos por canje',
            'Digitales': '📱 Ofertas app-first, push notifications, canje express',
            'Dormidos': '⏰ Reactivacion urgente, puntos bonus x2 por primera compra',
            'En Riesgo': '🚨 Retencion preventiva, alerta de puntos por vencer',
        }
        for cl in sorted(df.cluster_name.dropna().unique()):
            n = len(df[df.cluster_name==cl])
            st.markdown(f"**{cl}** ({n:,} clientes): {acciones.get(cl, 'Personalizar oferta')}")

# ══════════════════════════════════════════════════════════════
# VISTA 5: MODELO
# ══════════════════════════════════════════════════════════════
elif vista == "📈 Performance del Modelo":
    st.title("📈 Performance del Modelo Predictivo")
    st.markdown("**Modelo XGBoost entrenado sobre datos reales. Predice probabilidad de canje a 12 meses.**")

    if 'propensity_score' in df.columns and 'y' in df.columns:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            yb = (df.y>0).astype(int); pp = df.propensity_score.fillna(0)
            auc = roc_auc_score(yb, pp)

            c1,c2,c3 = st.columns(3)
            c1.metric("AUC", f"{auc:.4f}")
            c2.metric("Tasa base", f"{yb.mean()*100:.1f}%")
            c3.metric("Clientes test", f"{len(df):,}")

            # ROC
            fpr,tpr,_ = roc_curve(yb, pp)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f'Modelo (AUC={auc:.3f})',line=dict(color='steelblue',width=3)))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Aleatorio',line=dict(dash='dash',color='gray')))
            fig.update_layout(title='Curva ROC — Capacidad de discriminacion del modelo',
                xaxis_title='Tasa de falsos positivos',yaxis_title='Tasa de verdaderos positivos')
            st.plotly_chart(fig, use_container_width=True)

            # Lift
            st.subheader("Lift por decil — Eficiencia de targeting")
            st.markdown("*Si contactamos al top 10% segun el modelo, cuantos canjeadores capturamos?*")
            base=yb.mean()
            dft=df.copy(); dft['dec']=pd.qcut(dft.propensity_score.rank(method='first'),10,labels=range(1,11))
            ld=[]; cum=0; tot=yb.sum()
            for d in range(10,0,-1):
                s=dft[dft.dec==d]; r=(s.y>0).sum(); cum+=r
                ld.append({'Decil':d,'Tasa canje':r/len(s)*100,'Lift':r/len(s)/base,'Captura acumulada':cum/tot*100})
            ldf=pd.DataFrame(ld)

            fig2=make_subplots(specs=[[{"secondary_y":True}]])
            fig2.add_trace(go.Bar(x=ldf.Decil,y=ldf.Lift,name='Lift vs base',marker_color='steelblue'),secondary_y=False)
            fig2.add_trace(go.Scatter(x=ldf.Decil,y=ldf['Captura acumulada'],name='% canjeadores capturados',mode='lines+markers',marker_color='red'),secondary_y=True)
            fig2.update_layout(title='Lift por decil (10 = clientes con mayor propensity)',xaxis=dict(dtick=1))
            fig2.update_yaxes(title_text='Lift (veces sobre base)',secondary_y=False)
            fig2.update_yaxes(title_text='% canjeadores capturados',secondary_y=True)
            st.plotly_chart(fig2, use_container_width=True)

            top3_capture = ldf[ldf.Decil>=8]['Captura acumulada'].max()
            st.success(f"Contactando solo al **top 30%** de clientes (segun el modelo), capturamos el **{top3_capture:.0f}%** de todos los canjeadores.")
        except Exception as e:
            st.warning(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
# VISTA 6: INCREMENTALIDAD
# ══════════════════════════════════════════════════════════════
elif vista == "⚡ Incrementalidad":
    st.title("⚡ Incrementalidad — Impacto real del canje")
    st.markdown("""
    **Cuanto gasto incremental genera un cliente que canjea vs uno que no?**
    Comparamos la metodologia actual (GitLab) vs nuestro modelo causal (PSM).
    """)

    # Datos hardcodeados de las corridas anteriores
    gl = pd.DataFrame([
        {'Retail':'FALABELLA','Lift GitLab':28.1,'Lift PSM':21.2},
        {'Retail':'SODIMAC','Lift GitLab':25.0,'Lift PSM':3.9},
        {'Retail':'TOTTUS','Lift GitLab':37.0,'Lift PSM':23.5},
        {'Retail':'FCOM','Lift GitLab':19.3,'Lift PSM':28.9},
    ])

    c1,c2,c3 = st.columns(3)
    c1.metric("Lift GitLab (actual)", "+35.3%")
    c2.metric("Lift PSM (causal)", "+16.2%")
    c3.metric("Sobreestimacion", "2.2x")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=gl.Retail, y=gl['Lift GitLab'], name='GitLab (actual)', marker_color='gray'))
    fig.add_trace(go.Bar(x=gl.Retail, y=gl['Lift PSM'], name='PSM (causal)', marker_color='steelblue'))
    fig.update_layout(barmode='group', title='Lift gasto % por retail: metodo actual vs causal',
        yaxis_title='Lift gasto (%)', xaxis_title='Retail')
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Que significa esto?** El metodo actual (GitLab) reporta +35% de incrementalidad porque compara canjeadores vs no-canjeadores directamente.
    Pero los canjeadores ya eran clientes mas activos antes de canjear. Con PSM controlamos por esa autoseleccion y el lift real es +16%.
    Esto significa que de cada $100 de diferencia que GitLab reporta, solo $46 son realmente causados por el canje.
    """)

    # Uplift distribution
    if 'uplift_x' in df.columns:
        st.subheader("Distribucion del efecto individual (uplift)")
        fig2 = px.histogram(df, x='uplift_x', nbins=80, color='prioridad',
            color_discrete_map={'Alta':'green','Media':'orange','Baja':'gray','No contactar':'red'},
            title="A cuantos clientes vale la pena contactar?")
        fig2.update_layout(xaxis_title="Uplift individual ($ incremental estimado)", yaxis_title="Clientes")
        st.plotly_chart(fig2, use_container_width=True)

        pos_pct = (df.uplift_x>0).mean()*100
        st.success(f"**{pos_pct:.0f}%** de los clientes tienen uplift positivo — vale la pena contactarlos. El **{100-pos_pct:.0f}%** restante canjearia de todas formas o no responde a campañas.")

# ══════════════════════════════════════════════════════════════
# VISTA 7: MOTOR DE RECOMENDACIONES
# ══════════════════════════════════════════════════════════════
elif vista == "🎯 Motor de Recomendaciones":
    st.title("🎯 Motor de Recomendaciones")
    st.markdown("**Para cada cliente: que oferta, por que canal, con que urgencia**")

    # Prioridad
    st.subheader("Distribucion de prioridad de contacto")
    pri_df = df.groupby('prioridad').apply(lambda x: pd.Series({
        'Clientes':len(x), '%':len(x)/len(df)*100,
        'Tasa canje real':(x.y>0).mean()*100,
        'EV promedio':x.expected_value.mean() if 'expected_value' in x.columns else 0,
    })).reset_index()
    st.dataframe(pri_df.style.format({'Clientes':'{:,.0f}','%':'{:.1f}%','Tasa canje real':'{:.1f}%','EV promedio':'${:+,.0f}'}), use_container_width=True)

    fig = px.bar(pri_df, x='prioridad', y='Tasa canje real', color='prioridad',
        color_discrete_map={'Alta':'#4CAF50','Media':'#FFC107','Baja':'#9E9E9E','No contactar':'#F44336'},
        title="Tasa de canje real por prioridad asignada")
    fig.update_layout(yaxis_title="% que efectivamente canjeo", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Canal
    if 'canal' in df.columns:
        st.subheader("Canal recomendado")
        c1,c2 = st.columns(2)
        with c1:
            canal_dist = df.canal.value_counts()
            fig2 = px.pie(values=canal_dist.values, names=canal_dist.index, hole=0.4, title="Distribucion de canales")
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            if 'timing' in df.columns:
                timing_dist = df.timing.value_counts()
                fig3 = px.pie(values=timing_dist.values, names=timing_dist.index, hole=0.4, title="Urgencia de contacto")
                st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# VISTA 8: FICHA CLIENTE
# ══════════════════════════════════════════════════════════════
elif vista == "👤 Ficha Cliente":
    st.title("👤 Ficha de Cliente")
    cid = st.text_input("Buscar por cust_id", value=df_all.cust_id.iloc[0])
    c = df_all[df_all.cust_id==cid]
    if len(c)==0: st.error("Cliente no encontrado")
    else:
        r = c.iloc[0]
        st.subheader(f"Cliente: {cid}")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Categoria", str(r.get('tier','-')))
        c2.metric("Funnel", str(r.get('funnel_state_at_t0','-')))
        c3.metric("Cluster", str(r.get('cluster_name','-')))
        c4.metric("Prioridad", str(r.get('prioridad','-')))

        c5,c6,c7,c8 = st.columns(4)
        c5.metric("P(canje)", f"{r.propensity_score:.1%}" if 'propensity_score' in r.index else "-")
        c6.metric("CLV estimado", f"${r.clv_estimado:,.0f}" if 'clv_estimado' in r.index else "-")
        c7.metric("Gasto 12m", f"${r.monetary_total:,.0f}" if 'monetary_total' in r.index else "-")
        c8.metric("Stock puntos", f"{r.stock_points_at_t0:,.0f}" if 'stock_points_at_t0' in r.index else "-")

        st.subheader("Recomendacion")
        rec_cols = ['objetivo','accion','canal','timing']
        rec = {k: str(r.get(k, 'N/A')) for k in rec_cols}
        for k,v in rec.items():
            st.markdown(f"**{k.capitalize()}:** {v}")

# ══════════════════════════════════════════════════════════════
# VISTA 9: APERTURAS
# ══════════════════════════════════════════════════════════════
elif vista == "📋 Aperturas & Cross-tabs":
    st.title("📋 Aperturas detalladas")
    st.markdown("**Cruces entre todas las dimensiones del programa**")

    tab1,tab2,tab3,tab4 = st.tabs(["Quintil × Tier","Cluster × Funnel","Tipo × Prioridad","Quintil × Prioridad"])

    with tab1:
        if 'quintil_label' in df.columns and 'tier' in df.columns:
            st.subheader("Tasa de canje: Quintil de gasto × Categoria")
            ct = df.groupby(['quintil_label','tier']).apply(lambda x: (x.y>0).mean()*100).unstack(fill_value=0)
            fig = px.imshow(ct.values, x=ct.columns.tolist(), y=ct.index.tolist(), color_continuous_scale='RdYlGn',
                aspect='auto', title="Mapa de calor: % que canjea", labels=dict(color="%"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ct.style.format('{:.1f}%'), use_container_width=True)

    with tab2:
        if 'cluster_name' in df.columns and 'funnel_state_at_t0' in df.columns:
            st.subheader("Clientes: Cluster × Estado del funnel")
            ct2 = pd.crosstab(df.cluster_name, df.funnel_state_at_t0)
            fig2 = px.imshow(ct2.values, x=ct2.columns.tolist(), y=ct2.index.tolist(), color_continuous_scale='Blues',
                aspect='auto', title="Cantidad de clientes", labels=dict(color="N"))
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        if 'tipo_cliente' in df.columns and 'prioridad' in df.columns:
            st.subheader("Tipo de cliente × Prioridad")
            ct3 = pd.crosstab(df.tipo_cliente, df.prioridad, normalize='index')*100
            st.dataframe(ct3.style.format('{:.1f}%'), use_container_width=True)

    with tab4:
        if 'quintil_label' in df.columns and 'prioridad' in df.columns:
            st.subheader("Quintil × Prioridad")
            ct4 = pd.crosstab(df.quintil_label, df.prioridad, normalize='index')*100
            fig4 = px.bar(ct4.reset_index(), x='quintil_label',
                y=[c for c in ['Alta','Media','Baja','No contactar'] if c in ct4.columns],
                barmode='stack', title="Distribucion de prioridad por quintil de gasto")
            st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# VISTA 10: EXPORTS
# ══════════════════════════════════════════════════════════════
elif vista == "💾 Exportar Listas":
    st.title("💾 Exportar listas de clientes")

    st.subheader("Por prioridad de contacto")
    for pri in ['Alta','Media','Baja','No contactar']:
        s = df[df.prioridad==pri] if 'prioridad' in df.columns else pd.DataFrame()
        if len(s)==0: continue
        cols = [c for c in ['cust_id','tier','funnel_state_at_t0','cluster_name','propensity_score','expected_value','prioridad','canal','timing'] if c in s.columns]
        csv = s[cols].to_csv(index=False)
        st.download_button(f"📥 {pri} ({len(s):,} clientes)", csv, f"lista_{pri.lower().replace(' ','_')}.csv")

    st.subheader("Scoring completo")
    cols_all = [c for c in df.columns if c not in ['dec']]
    csv_all = df[cols_all].to_csv(index=False)
    st.download_button("📥 Descargar scoring completo", csv_all, "scoring_completo.csv")
