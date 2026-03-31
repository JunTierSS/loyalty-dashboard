"""🎯 Loyalty Intelligence System — CMR Puntos
Modelo binario 1 mes: P(canje en proximo mes)
Si canjea + nunca canjeo antes = Canjeador Nuevo
Si canjea + ya canjeo antes = Recurrente
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json,warnings

# ═══════ CONSTANTES ═══════
TIER_ORDER = ['Entrada','Fan','Premium','Elite']
FUNNEL_ORDER = ['Inscrito','Participante','Posibilidad Canje','Canjeador','Recurrente','Fuga']
PRIORIDAD_ORDER = ['Urgente','Alta','Media','Baja','No contactar']
QUINTIL_ORDER = ['Q1','Q2','Q3','Q4','Q5']
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Loyalty Intelligence",page_icon="🎯",layout="wide")

# ═══════ CARGA ═══════
@st.cache_data(ttl=3600,show_spinner="Cargando datos...")
def load():
    df=pd.read_csv('data_scored.csv.gz',compression='gzip')
    tmp=pd.read_csv('data_temporal.csv')
    fevo=pd.read_csv('data_funnel_evo.csv')
    met=json.load(open('data_metrics.json'))
    try: incr=json.load(open('data_incrementalidad.json'))
    except: incr={}
    met['_incr']=incr
    str_cols=['cust_id','tier','gender','city','dominant_retailer','funnel_state_at_t0',
              'status','cluster_name','prioridad','canal','timing','objetivo','accion',
              'tipo_cliente','quintil_label','target_label','target_label_12m',
              'rfm_segment','t0','tipo_predicho','tipo_canje']
    for c in df.columns:
        if c not in str_cols:
            try: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
            except: pass
    if 'quintil_label' not in df.columns and 'quintil_gasto' in df.columns:
        df['quintil_label']=df['quintil_gasto'].map({1:'Q1',2:'Q2',3:'Q3',4:'Q4',5:'Q5'})
    if 'breakage' not in df.columns: df['breakage']=(1-df['redeem_rate'].clip(0,1))
    if 'clv_estimado' not in df.columns: df['clv_estimado']=df['revenue_post_12m']*1.5
    if 'y_1m' in df.columns: df['y_target']=df['y_1m']
    else: df['y_target']=(df['y']>0).astype(int)
    if 'p_canje_1m' in df.columns: df['prob']=df['p_canje_1m']
    elif 'propensity_score' in df.columns: df['prob']=df['propensity_score']
    else: df['prob']=0
    # Ensure categoricals are ordered for plotly
    df['tier']=pd.Categorical(df['tier'],categories=TIER_ORDER,ordered=True)
    df['funnel_state_at_t0']=pd.Categorical(df['funnel_state_at_t0'],categories=FUNNEL_ORDER,ordered=True)
    if 'quintil_label' in df.columns:
        df['quintil_label']=pd.Categorical(df['quintil_label'],categories=QUINTIL_ORDER,ordered=True)
    if 'prioridad' in df.columns:
        df['prioridad']=pd.Categorical(df['prioridad'],categories=PRIORIDAD_ORDER,ordered=True)
    return df,tmp,fevo,met

A,T,FE,M=load()

# ═══════ SIDEBAR ═══════
st.sidebar.title("🎯 Loyalty Intelligence")
st.sidebar.caption(f"CMR Puntos | {M['n_clientes']:,} clientes | Modelo binario {M.get('horizonte','1 mes')}")
VIEWS = ["🏠 Resumen","💰 CLV & Revenue","📊 RFM","🔄 Funnel","🧩 Segmentos",
         "📈 Modelo","⚡ Incrementalidad","🔮 Prediccion Mes",
         "🎯 Motor Decision","🎚️ Simulador","👤 Ficha Cliente","📋 Aperturas","💾 Exportar"]
V=st.sidebar.radio("",VIEWS)
st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros**")
def safe_opts(col):
    if col not in A.columns: return []
    vals = [x for x in A[col].dropna().unique() if str(x) not in ['','Desconocido','UNKNOWN','nan']]
    # Use order if available
    order_map = {'tier':TIER_ORDER,'funnel_state_at_t0':FUNNEL_ORDER,'prioridad':PRIORIDAD_ORDER,'quintil_label':QUINTIL_ORDER}
    if col in order_map:
        order=order_map[col]
        return [x for x in order if x in vals]
    return sorted(vals)

tiers=st.sidebar.multiselect("Categoria",safe_opts('tier'),default=safe_opts('tier'))
if 'cluster_name' in A.columns: cls=st.sidebar.multiselect("Cluster",safe_opts('cluster_name'),default=safe_opts('cluster_name'))
else: cls=[]
fns=st.sidebar.multiselect("Funnel",safe_opts('funnel_state_at_t0'),default=safe_opts('funnel_state_at_t0'))
qts=st.sidebar.multiselect("Quintil",safe_opts('quintil_label'),default=safe_opts('quintil_label'))
prs=st.sidebar.multiselect("Prioridad",safe_opts('prioridad'),default=safe_opts('prioridad'))

def fl(d):
    m=pd.Series(True,index=d.index)
    if tiers: m&=d.tier.isin(tiers)
    if cls and 'cluster_name' in d.columns: m&=d.cluster_name.isin(cls)
    if fns: m&=d.funnel_state_at_t0.isin(fns)
    if qts and 'quintil_label' in d.columns: m&=d.quintil_label.isin(qts)
    if prs and 'prioridad' in d.columns: m&=d.prioridad.isin(prs)
    return d[m]
df=fl(A)
st.sidebar.caption(f"📊 {len(df):,} registros")

def fmt(v):
    if abs(v)>=1e9: return f"${v/1e9:,.1f}B"
    if abs(v)>=1e6: return f"${v/1e6:,.0f}M"
    return f"${v:,.0f}"

# Shared category_orders for plotly
CAT_ORDERS = {
    'tier': TIER_ORDER,
    'funnel_state_at_t0': FUNNEL_ORDER,
    'prioridad': PRIORIDAD_ORDER,
    'quintil_label': QUINTIL_ORDER,
}

# ══════════════════════════════════════════════════════════════
if V=="🏠 Resumen":
    st.title("🏠 Resumen Ejecutivo")
    st.markdown(f"""
> **Sistema de inteligencia** para {M['n_clientes']:,} clientes reales de CMR Puntos.
> Modelo XGBoost binario predice P(canje proximo mes) con **AUC {M['auc']:.3f}**.
> Los datos cubren 27 snapshots mensuales (Ene-2023 a Mar-2025).
    """)
    k1,k2,k3,k4,k5,k6=st.columns(6)
    k1.metric("Clientes",f"{df.cust_id.nunique():,}")
    k2.metric("Tasa canje 1m",f"{df.y_target.mean()*100:.1f}%")
    k3.metric("AUC",f"{M['auc']:.3f}")
    k4.metric("Canjearan (pred)",f"{(df.prob>M.get('threshold',0.5)).sum():,}")
    k5.metric("Gasto Prom",fmt(df.monetary_total.mean()))
    urg_pct = (df.prioridad=='Urgente').mean()*100 if 'prioridad' in df.columns else 0
    k6.metric("% Urgente",f"{urg_pct:.1f}%")
    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("💡 Oportunidad: Activar primeros canjes")
        nunca=df[(df.has_redeemed_before_t0==False)&(df.stock_points_at_t0>=1000)]
        prob_alta=nunca[nunca.prob>0.2]
        st.metric("Nunca canjearon + tienen puntos",f"{len(nunca):,}")
        st.metric("Con P(canje)>20%",f"{len(prob_alta):,}")
        rev=len(prob_alta)*df[df.y_target==1].revenue_post_12m.mean()/12 if (df.y_target==1).any() else 0
        st.info(f"**{len(prob_alta):,}** clientes listos para su primer canje. Revenue potencial proximo mes: **{fmt(rev)}**")
    with c2:
        st.subheader("⚠️ Fuga & Urgencia")
        fuga=df[df.funnel_state_at_t0=='Fuga']
        urg=df[df.prioridad=='Urgente'] if 'prioridad' in df.columns else pd.DataFrame()
        st.metric("En Fuga",f"{len(fuga):,}")
        st.metric("Prioridad Urgente",f"{len(urg):,}")
        st.warning(f"**{len(fuga):,}** clientes en fuga. Requieren campana inmediata de reactivacion.")

    st.subheader("Tasa canje por categoria")
    td=df.groupby('tier',observed=True).apply(lambda x: pd.Series({'Canjea 1m':x.y_target.mean()*100,'P(canje) prom':x.prob.mean()*100})).reset_index()
    fig=px.bar(td,x='tier',y='Canjea 1m',color='P(canje) prom',color_continuous_scale='RdYlGn',
               title="% que canjea en 1 mes por categoria",category_orders=CAT_ORDERS)
    fig.update_layout(yaxis_title="%",xaxis_title="Categoria")
    st.plotly_chart(fig,use_container_width=True)

    if 't0_label' in T.columns:
        st.subheader("Evolucion temporal")
        col_y='tasa_canje_1m' if 'tasa_canje_1m' in T.columns else 'tasa_canje'
        fig2=px.line(T,x='t0_label',y=col_y,markers=True,title="Tasa de canje mensual")
        fig2.update_layout(yaxis_title="%",xaxis_title="Mes")
        st.plotly_chart(fig2,use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="💰 CLV & Revenue":
    st.title("💰 CLV & Revenue")
    st.markdown("""
> **CLV (Customer Lifetime Value)** estima cuanto valor genera cada cliente a futuro.
> Calculado como `revenue_post_12m × 1.5` (factor de retencion).
> Permite comparar el valor de un canjeador vs no canjeador y justificar inversion en campanas.
    """)
    clv_c=df[df.y_target==1].clv_estimado.mean() if (df.y_target==1).any() else 0
    clv_n=df[df.y_target==0].clv_estimado.mean() if (df.y_target==0).any() else 0
    ratio=clv_c/clv_n if clv_n>0 else 0
    c1,c2,c3=st.columns(3)
    c1.metric("CLV Canjeador",fmt(clv_c)); c2.metric("CLV No canjeador",fmt(clv_n)); c3.metric("Multiplicador",f"{ratio:.1f}x")
    st.success(f"Cada cliente que canjea genera **{ratio:.1f}x mas valor**. Activar 100 clientes = {fmt(100*(clv_c-clv_n))} en CLV adicional.")

    st.subheader("CLV por categoria")
    ct=df.groupby('tier',observed=True).apply(lambda x: pd.Series({
        'CLV Canjea':x[x.y_target==1].clv_estimado.mean() if (x.y_target==1).any() else 0,
        'CLV No canjea':x[x.y_target==0].clv_estimado.mean() if (x.y_target==0).any() else 0
    })).reset_index()
    fig=px.bar(ct.melt(id_vars='tier'),x='tier',y='value',color='variable',barmode='group',
               title="CLV por categoria y comportamiento",labels={'value':'CLV ($)','tier':'Categoria'},
               category_orders=CAT_ORDERS)
    st.plotly_chart(fig,use_container_width=True)

    if 'revenue_esperado_1m' in df.columns:
        st.subheader("Revenue esperado proximo mes")
        rev_tier=df.groupby('tier',observed=True).revenue_esperado_1m.sum().reset_index()
        fig2=px.bar(rev_tier,x='tier',y='revenue_esperado_1m',title="Revenue esperado por categoria",
                    labels={'revenue_esperado_1m':'Revenue ($)'},category_orders=CAT_ORDERS)
        st.plotly_chart(fig2,use_container_width=True)
        st.metric("Revenue total esperado proximo mes",fmt(df.revenue_esperado_1m.sum()))

# ══════════════════════════════════════════════════════════════
elif V=="📊 RFM":
    st.title("📊 Analisis RFM")
    st.markdown("""
> **RFM** segmenta clientes por 3 dimensiones: **R**ecency (dias desde ultima compra),
> **F**requency (compras en 12m), **M**onetary (gasto en 12m). Cada dimension se puntua 1-5.
>
> | Segmento | Descripcion | Accion |
> |---|---|---|
> | **Champions** | Recientes + frecuentes + alto gasto | Retener y premiar |
> | **Loyal** | Buenos en las 3 dimensiones | Mantener engagement |
> | **En Riesgo** | Fueron buenos pero se estan yendo | Campana urgente |
> | **Perdidos** | Inactivos, baja frecuencia | Reactivar o soltar |
> | **Nuevos** | Recientes pero poca frecuencia | Desarrollar habito |
    """)
    if 'rfm_segment' in df.columns:
        rfm=df.groupby('rfm_segment').apply(lambda x: pd.Series({
            'Clientes':len(x),'%':len(x)/len(df)*100,
            'Tasa canje':x.y_target.mean()*100,
            'Recency (dias)':x.recency_days.mean(),
            'Frecuencia':x.frequency_total.mean(),
            'Gasto prom':x.monetary_total.mean(),
            'CLV':x.clv_estimado.mean()
        })).reset_index().sort_values('CLV',ascending=False)
        st.dataframe(rfm.style.format({'Clientes':'{:,.0f}','%':'{:.1f}%','Tasa canje':'{:.1f}%',
            'Recency (dias)':'{:.0f}','Frecuencia':'{:.1f}','Gasto prom':'${:,.0f}','CLV':'${:,.0f}'}),
            use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            fig=px.pie(rfm,names='rfm_segment',values='Clientes',hole=.4,title="Distribucion RFM")
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            samp=df.sample(min(2000,len(df)),random_state=42)
            fig2=px.scatter(samp,x='recency_days',y='frequency_total',
                           size=np.clip(samp.monetary_total,0,samp.monetary_total.quantile(0.95)),
                           color='rfm_segment',title="Recency x Frecuencia (tamano=gasto)",opacity=0.5)
            fig2.update_layout(xaxis_title="Dias desde ultima compra (menor=mejor)",yaxis_title="N compras 12m")
            st.plotly_chart(fig2,use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="🔄 Funnel":
    st.title("🔄 Funnel de Canje")
    st.markdown("""
> El funnel representa el viaje del cliente dentro del programa de fidelizacion.
> Cada estado tiene reglas de transicion basadas en actividad y canjes:
>
> **Inscrito** (registro) → **Participante** (al menos 1 compra) → **Posibilidad Canje** (≥1000 puntos)
> → **Canjeador** (realiza 1 canje) → **Recurrente** (2+ canjes en 12m)
> ↘ **Fuga** (fue canjeador/recurrente pero >12 meses sin canjear)
>
> *El objetivo es mover clientes hacia la derecha y evitar la Fuga.*
    """)
    if 't0_label' in FE.columns and len(FE)>10:
        fig=px.area(FE,x='t0_label',y='n',color='funnel_state_at_t0',
                    category_orders={'funnel_state_at_t0':FUNNEL_ORDER},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    title="Evolucion del funnel — clientes en cada estado por mes")
        fig.update_layout(yaxis_title="Clientes",xaxis_title="Mes",legend_title="Estado")
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.warning("Datos de funnel temporal no disponibles o insuficientes.")

    c1,c2=st.columns(2)
    with c1:
        fd=df.groupby('funnel_state_at_t0',observed=True).size().reset_index(name='n')
        fig2=px.pie(fd,values='n',names='funnel_state_at_t0',hole=.4,title="Distribucion actual",
                    category_orders=CAT_ORDERS)
        st.plotly_chart(fig2,use_container_width=True)
    with c2:
        fr=df.groupby('funnel_state_at_t0',observed=True).apply(lambda x: x.y_target.mean()*100).reset_index(name='Tasa 1m')
        fig3=px.bar(fr,x='funnel_state_at_t0',y='Tasa 1m',color='Tasa 1m',
                    color_continuous_scale='RdYlGn',title="P(canje proximo mes) por estado",
                    category_orders=CAT_ORDERS)
        fig3.update_layout(yaxis_title="% canjea",xaxis_title="Estado")
        st.plotly_chart(fig3,use_container_width=True)

    big=df.funnel_state_at_t0.value_counts()
    if len(big)>0:
        st.info(f"🔍 **Cuello de botella:** {big.idxmax()} ({big.max():,} clientes, {big.max()/len(df)*100:.0f}%). "
                f"Oportunidad: mover a estos clientes al siguiente estado.")

# ══════════════════════════════════════════════════════════════
elif V=="🧩 Segmentos":
    st.title("🧩 Segmentos (K-Means)")
    st.markdown("""
> Clustering K-Means sobre 5 features normalizadas: frecuencia, gasto, recency, puntos, redeem_rate.
> Se asignan nombres de arquetipos via Hungarian Algorithm (matching optimo).
> Los clusters permiten personalizar campanas: no es lo mismo un Heavy User que un Dormido.
    """)
    if 'cluster_name' in df.columns:
        cl=df.groupby('cluster_name').apply(lambda x: pd.Series({
            'N':len(x),'%':len(x)/len(df)*100,'Tasa 1m':x.y_target.mean()*100,
            'P(canje)':x.prob.mean(),'CLV':x.clv_estimado.mean(),'Gasto prom':x.monetary_total.mean()
        })).reset_index().sort_values('CLV',ascending=False)
        st.dataframe(cl.style.format({'N':'{:,.0f}','%':'{:.1f}%','Tasa 1m':'{:.1f}%',
            'P(canje)':'{:.3f}','CLV':'${:,.0f}','Gasto prom':'${:,.0f}'}),use_container_width=True)

        acc={'Heavy Users':'🏆 Upgrade tier, experiencia exclusiva, early access',
             'Exploradores':'🔍 Ofertas multi-retailer, cross-sell entre tiendas',
             'Cazadores de Canje':'🎁 Descuento 20% puntos, catalogo personalizado',
             'Digitales':'📱 Push notifications, canje express en app',
             'Dormidos':'⏰ Puntos x2, oferta de bienvenida, email personalizado',
             'En Riesgo':'🚨 Alerta puntos por vencer, llamada directa'}
        st.subheader("Accion recomendada por cluster")
        for cn in sorted(df.cluster_name.dropna().unique()):
            n=len(df[df.cluster_name==cn])
            st.markdown(f"**{cn}** ({n:,}): {acc.get(cn,'Personalizar segun perfil')}")

        # Cluster × tier heatmap
        st.subheader("Cluster × Categoria")
        ct_cl = pd.crosstab(df.cluster_name, df.tier)
        fig=px.imshow(ct_cl.values,x=[str(c) for c in ct_cl.columns],y=ct_cl.index.tolist(),
                      color_continuous_scale='Blues',aspect='auto',title="N clientes: Cluster × Categoria")
        st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="📈 Modelo":
    st.title("📈 Modelo Predictivo")
    threshold = M.get('threshold', 0.5)
    st.markdown(f"""
> **XGBoost binario** — predice P(canje) para cada cliente.
>
> **Metodologia:**
> - Entrenamos con **{M['n_clientes']:,} clientes reales** del programa CMR Puntos
> - **~48 features**: RFM, puntos (stock, vencimiento, velocidad), funnel, tier, actividad digital, estacionalidad
> - **Train:** t0s de Ene-2023 a Sep-2024 | **Test:** t0s de Ene-2025 a Mar-2025
> - **Modelo:** {M.get('modelo','XGBoost binario')} con scale_pos_weight={M.get('scale_pos_weight','auto')}
> - **Threshold optimo:** {threshold:.2f} (optimizado para maximizar F1)
> - **Post-clasificacion:** si canjea y nunca habia canjeado = Canjeador Nuevo; si ya habia canjeado = Recurrente
    """)
    try:
        from sklearn.metrics import roc_auc_score,roc_curve,f1_score,precision_score,recall_score,accuracy_score,confusion_matrix
        yb=df.y_target.astype(int); pp=df.prob
        auc=roc_auc_score(yb,pp)
        yp=(pp>threshold).astype(int)
        acc_s=accuracy_score(yb,yp); f1=f1_score(yb,yp); prec=precision_score(yb,yp,zero_division=0); rec=recall_score(yb,yp)

        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("AUC",f"{auc:.4f}"); c2.metric("Accuracy",f"{acc_s:.1%}"); c3.metric("F1",f"{f1:.3f}")
        c4.metric("Precision",f"{prec:.3f}"); c5.metric("Recall",f"{rec:.3f}"); c6.metric("Tasa base",f"{yb.mean()*100:.1f}%")

        st.markdown(f"""
> **Interpretacion de metricas:**
> - **AUC={auc:.3f}**: Excelente discriminacion entre canjeadores y no canjeadores
> - **Precision={prec:.1%}**: De los predichos como canjeadores, {prec:.0%} efectivamente canjea
> - **Recall={rec:.1%}**: De los que canjean, el modelo identifica al {rec:.0%}
> - **F1={f1:.3f}**: Media armonica entre Precision y Recall
> - **Accuracy={acc_s:.1%}**: % de predicciones correctas (inflado por la clase mayoritaria)
        """)

        # Confusion matrix
        st.subheader("Matriz de Confusion")
        cm=confusion_matrix(yb,yp)
        labels_cm = [['Verdadero Negativo','Falso Positivo'],['Falso Negativo','Verdadero Positivo']]
        text_cm = [[f"{labels_cm[i][j]}<br>{cm[i,j]:,}" for j in range(2)] for i in range(2)]
        fig_cm=go.Figure(data=go.Heatmap(
            z=cm, x=['Pred: No canjea','Pred: Canjea'], y=['Real: No canjea','Real: Canjea'],
            text=text_cm, texttemplate="%{text}", colorscale='Blues', showscale=False
        ))
        fig_cm.update_layout(title='Prediccion vs Realidad',height=400)
        st.plotly_chart(fig_cm,use_container_width=True)
        st.caption(f"Total: {cm.sum():,} | No canjea real: {cm[0].sum():,} | Canjea real: {cm[1].sum():,}")

        # ROC curve
        fpr,tpr,_=roc_curve(yb,pp)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f'AUC={auc:.3f}',line=dict(width=3,color='steelblue')))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(dash='dash',color='gray'),name='Aleatorio'))
        fig.update_layout(title='Curva ROC',xaxis_title='Tasa falsos positivos',yaxis_title='Tasa verdaderos positivos')
        st.plotly_chart(fig,use_container_width=True)

        # Lift chart
        st.subheader("Lift — Eficiencia de targeting")
        st.markdown("> Si solo podemos contactar al X% de clientes, cuantos canjeadores capturamos?")
        base=yb.mean(); dt=df.copy(); dt['dec']=pd.qcut(dt.prob.rank(method='first'),10,labels=range(1,11))
        ld=[]; cum=0; tot=yb.sum()
        for d in range(10,0,-1):
            s=dt[dt.dec==d]; r=int(s.y_target.sum()); cum+=r
            ld.append({'Decil':d,'N':len(s),'Canjeadores':r,'Tasa':r/len(s)*100,
                       'Lift':r/len(s)/base if base>0 else 0,'Captura acum':cum/tot*100 if tot>0 else 0})
        ldf=pd.DataFrame(ld)

        fig2=make_subplots(specs=[[{"secondary_y":True}]])
        fig2.add_trace(go.Bar(x=ldf.Decil,y=ldf.Lift,name='Lift vs base',marker_color='steelblue'),secondary_y=False)
        fig2.add_trace(go.Scatter(x=ldf.Decil,y=ldf['Captura acum'],name='% capturado acum',mode='lines+markers',marker_color='red'),secondary_y=True)
        fig2.update_layout(title='Contactando al top X%, cuantos canjeadores capturamos?',xaxis=dict(dtick=1))
        fig2.update_yaxes(title_text='Lift (veces sobre base)',secondary_y=False)
        fig2.update_yaxes(title_text='% canjeadores capturados',secondary_y=True)
        st.plotly_chart(fig2,use_container_width=True)

        st.dataframe(ldf.style.format({'N':'{:,.0f}','Canjeadores':'{:,.0f}','Tasa':'{:.1f}%','Lift':'{:.2f}x','Captura acum':'{:.1f}%'}),use_container_width=True)

        t30=ldf[ldf.Decil>=8]['Captura acum'].max() if len(ldf)>0 else 0
        st.success(f"Contactando al **top 30%** capturamos el **{t30:.0f}%** de los canjeadores del proximo mes.")

        # Metricas por tier
        st.subheader("AUC por categoria")
        auc_tier = []
        for t in TIER_ORDER:
            s=df[df.tier==t]
            if len(s)>30 and s.y_target.sum()>5:
                try: auc_tier.append({'Categoria':t,'N':len(s),'AUC':roc_auc_score(s.y_target.astype(int),s.prob),'Tasa canje':s.y_target.mean()*100})
                except: pass
        if auc_tier:
            st.dataframe(pd.DataFrame(auc_tier).style.format({'N':'{:,.0f}','AUC':'{:.4f}','Tasa canje':'{:.1f}%'}),use_container_width=True)

    except Exception as e: st.warning(f"Error calculando metricas: {e}")

# ══════════════════════════════════════════════════════════════
elif V=="⚡ Incrementalidad":
    st.title("⚡ Incrementalidad del Programa de Canje")
    st.markdown("""
> **Pregunta clave:** Cuanto gasto **adicional** genera el canje? O los que canjean simplemente ya gastaban mas?
>
> **Dos metodologias:**
> - **GitLab (actual):** Compara canjeadores vs potenciales directamente, controlando por quintil de gasto previo.
>   Sobreestima porque los canjeadores ya eran mas activos (autoseleccion).
> - **PSM (causal):** Propensity Score Matching — empareja clientes similares (misma probabilidad de canjear)
>   y compara gasto. Mide el efecto **real** del canje.
>
> **Periodos GitLab (3 periodos, 12m cada uno):**
> - PREV: Mar-2022 a Mar-2023 (gasto historico)
> - SEL: Mar-2023 a Mar-2024 (periodo de seleccion — quien canjeo aqui?)
> - POST: Mar-2024 a Mar-2025 (gasto posterior — este es el que se compara)
    """)

    # Datos reales de la corrida con filtros exactos GitLab
    incr = M.get('_incr', {})
    retailers_incr = ['FALABELLA','SODIMAC','TOTTUS','FCOM']
    gl_rows = []
    for ret in retailers_incr:
        if ret in incr:
            gl_rows.append({'Retail':ret,'Lift %':incr[ret]['lift_pct'],
                           'N Canj':incr[ret]['n_canjeadores'],'N Pot':incr[ret]['n_potenciales']})
    if 'TOTAL' in incr:
        total_lift = incr['TOTAL']['lift_pct']
    else:
        total_lift = 0

    if gl_rows:
        gl=pd.DataFrame(gl_rows)
        c1,c2,c3=st.columns(3)
        c1.metric("Lift GitLab (TOTAL)",f"+{total_lift:.1f}%")
        c2.metric("N Canjeadores",f"{incr.get('TOTAL',{}).get('n_canjeadores',0):,}")
        c3.metric("N Potenciales",f"{incr.get('TOTAL',{}).get('n_potenciales',0):,}")

        fig=px.bar(gl,x='Retail',y='Lift %',color='Lift %',color_continuous_scale='RdYlGn',
                   title='Lift gasto % por retail (metodologia GitLab exacta)',
                   text='Lift %')
        fig.update_traces(texttemplate='%{text:+.1f}%', textposition='outside')
        fig.update_layout(yaxis_title='Lift gasto %',xaxis_title='Retail',showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

    # Tablas detalladas por quintil
    st.subheader("📊 Tablas por quintil — metodologia GitLab exacta")
    st.markdown("""
> **Filtros aplicados (identicos a produccion):**
> 1. Solo Canjeadores y Potenciales (excluye Acumuladores)
> 2. GASTO_PREV > 0 AND GASTO_POST > 0 (por retail)
> 3. Excluir top 1% de gasto PRE
> 4. Excluir top 1% de gasto POST
> 5. Primer canje NO puede ser posterior al periodo SEL
>
> **Lift** = (Gasto prom Canjeador - Gasto prom Potencial) / Gasto prom Potencial
    """)

    tab_names = ["TOTAL"] + retailers_incr
    tab_r = st.tabs(tab_names)
    for idx, ret in enumerate(tab_names):
        with tab_r[idx]:
            if ret in incr and 'quintiles' in incr[ret]:
                q_df = pd.DataFrame(incr[ret]['quintiles'])
                st.dataframe(q_df.style.format({
                    'N Canj':'{:,.0f}','N Pot':'{:,.0f}',
                    'Gasto prom Cj':'${:,.0f}','Gasto prom Pt':'${:,.0f}',
                    'Gasto total Cj':'${:,.0f}','Gasto total Pt':'${:,.0f}',
                    'Lift %':'{:+.1f}%'
                }),use_container_width=True)
                st.caption(f"Lift total ponderado {ret}: **{incr[ret]['lift_pct']:+.1f}%** | CJ={incr[ret]['n_canjeadores']:,} PT={incr[ret]['n_potenciales']:,}")
            else:
                st.info(f"Sin datos para {ret}")

    # Uplift distribution
    if 'uplift_x' in df.columns:
        st.subheader("Distribucion de uplift individual (T-Learner)")
        st.markdown("> El uplift estima el **efecto causal** de contactar a cada cliente. Positivo = vale la pena contactar.")
        p1,p99=df.uplift_x.quantile(0.02),df.uplift_x.quantile(0.98)
        dc=df[(df.uplift_x>=p1)&(df.uplift_x<=p99)]
        fig2=px.histogram(dc,x='uplift_x',nbins=50,color='prioridad',
                          color_discrete_map={'Alta':'green','Media':'orange','Baja':'gray','No contactar':'red','Urgente':'darkred'},
                          title="Efecto individual: a quien le sirve que lo contactemos?",
                          category_orders=CAT_ORDERS)
        fig2.update_layout(xaxis_title="Uplift ($) — positivo = vale la pena contactar",yaxis_title="Clientes")
        st.plotly_chart(fig2,use_container_width=True)
        pos_pct=(dc.uplift_x>0).mean()*100
        st.metric("% clientes con uplift positivo",f"{pos_pct:.1f}%")

# ══════════════════════════════════════════════════════════════
elif V=="🔮 Prediccion Mes":
    st.title("🔮 Que esperamos el proximo mes")
    st.markdown(f"""
> **Modelo binario 1 mes:** Para cada uno de los {len(df):,} clientes, el modelo calcula P(canje en 30 dias).
>
> - **Canjearan:** clientes con mas del 50% de probabilidad de canjear
> - **Canjeador Nuevo:** nunca canjeo antes + P(canje)>20% (umbral menor porque queremos capturar activaciones)
> - **Recurrente:** ya canjeo antes + P>50%
> - **Revenue esperado** = Σ P(canje_i) × gasto_mensual_i para todos los clientes
> - **Puntos a canjear** = Σ P(canje_i) × promedio_puntos_por_canje
    """)
    pred_threshold = M.get('threshold', 0.5)
    act_threshold = max(0.2, pred_threshold * 0.4)  # lower threshold for activation
    n_canjean=int((df.prob>pred_threshold).sum())
    n_nuevos=int(((df.prob>act_threshold)&(df.has_redeemed_before_t0==False)).sum()) if 'has_redeemed_before_t0' in df.columns else 0
    n_recur=int(((df.prob>pred_threshold)&(df.has_redeemed_before_t0==True)).sum()) if 'has_redeemed_before_t0' in df.columns else 0
    rev_esp=df.revenue_esperado_1m.sum() if 'revenue_esperado_1m' in df.columns else 0
    pts_esp=df.puntos_esperados_canje.sum() if 'puntos_esperados_canje' in df.columns else 0
    pts_vencer=df.exp_points_current_at_t0.sum() if 'exp_points_current_at_t0' in df.columns else 0

    c1,c2,c3=st.columns(3)
    c1.metric("🎯 Canjearan",f"{n_canjean:,}")
    c2.metric("🆕 Canjeadores nuevos",f"{n_nuevos:,}")
    c3.metric("🔁 Recurrentes",f"{n_recur:,}")
    c4,c5,c6=st.columns(3)
    c4.metric("💰 Revenue esperado",fmt(rev_esp))
    c5.metric("🎁 Puntos a canjear",f"{pts_esp:,.0f}")
    c6.metric("⏰ Puntos por vencer",f"{pts_vencer:,.0f}")

    # Tabla Quintil × Categoria
    st.subheader("📊 Apertura: Quintil × Categoria")
    st.markdown("> Cuantos clientes canjearan en cada segmento de gasto × categoria")

    tabs_pred = st.tabs(["Tabla general"] + TIER_ORDER)
    with tabs_pred[0]:
        if 'quintil_label' in df.columns:
            pred_q = df.groupby(['quintil_label','tier'],observed=True).apply(lambda x: pd.Series({
                'N clientes':len(x),
                'P(canje) prom':x.prob.mean()*100,
                'Canjearan':(x.prob>M.get('threshold',0.5)).sum(),
                'Nuevos':((x.prob>0.2)&(x.has_redeemed_before_t0==False)).sum() if 'has_redeemed_before_t0' in x.columns else 0,
                'Revenue esp':x.revenue_esperado_1m.sum() if 'revenue_esperado_1m' in x.columns else 0,
            })).reset_index()
            pivot = pred_q.pivot_table(index='quintil_label',columns='tier',values='Canjearan',fill_value=0,observed=True)
            # Reorder columns
            pivot = pivot.reindex(columns=[c for c in TIER_ORDER if c in pivot.columns])
            st.markdown("**Canjeadores esperados (P>50%) por Quintil × Categoria**")
            st.dataframe(pivot.style.format('{:.0f}'),use_container_width=True)

            pivot_rev = pred_q.pivot_table(index='quintil_label',columns='tier',values='Revenue esp',fill_value=0,observed=True)
            pivot_rev = pivot_rev.reindex(columns=[c for c in TIER_ORDER if c in pivot_rev.columns])
            st.markdown("**Revenue esperado ($) por Quintil × Categoria**")
            st.dataframe(pivot_rev.style.format('${:,.0f}'),use_container_width=True)

            pivot_p = pred_q.pivot_table(index='quintil_label',columns='tier',values='P(canje) prom',fill_value=0,observed=True)
            pivot_p = pivot_p.reindex(columns=[c for c in TIER_ORDER if c in pivot_p.columns])
            fig=px.imshow(pivot_p.values,x=[str(c) for c in pivot_p.columns],y=pivot_p.index.tolist(),
                          color_continuous_scale='RdYlGn',aspect='auto',
                          title="P(canje) promedio: Quintil × Categoria",labels=dict(color="%"))
            st.plotly_chart(fig,use_container_width=True)

    for idx, tier_name in enumerate(TIER_ORDER):
        with tabs_pred[idx+1]:
            tier_df = df[df.tier==tier_name]
            if len(tier_df)==0:
                st.info(f"Sin datos para {tier_name}")
                continue
            if 'quintil_label' in tier_df.columns:
                t_q = tier_df.groupby('quintil_label',observed=True).apply(lambda x: pd.Series({
                    'N clientes':len(x),
                    'P(canje) prom':f"{x.prob.mean()*100:.1f}%",
                    'Canjearan':int((x.prob>M.get('threshold',0.5)).sum()),
                    'Nuevos':int(((x.prob>0.2)&(x.has_redeemed_before_t0==False)).sum()) if 'has_redeemed_before_t0' in x.columns else 0,
                    'Revenue esp':fmt(x.revenue_esperado_1m.sum()) if 'revenue_esperado_1m' in x.columns else '$0',
                    'Pts por vencer':f"{x.exp_points_current_at_t0.sum():,.0f}" if 'exp_points_current_at_t0' in x.columns else '0',
                })).reset_index()
                st.dataframe(t_q,use_container_width=True)

    # Apertura por categoria (simple)
    st.subheader("Resumen por categoria")
    pred=df.groupby('tier',observed=True).apply(lambda x: pd.Series({
        'Total':len(x),
        'Canjearan':int((x.prob>M.get('threshold',0.5)).sum()),
        'Nuevos (P>20%)':int(((x.prob>0.2)&(x.has_redeemed_before_t0==False)).sum()) if 'has_redeemed_before_t0' in x.columns else 0,
        'Revenue esp':x.revenue_esperado_1m.sum() if 'revenue_esperado_1m' in x.columns else 0,
        'Pts por vencer':x.exp_points_current_at_t0.sum() if 'exp_points_current_at_t0' in x.columns else 0,
    })).reset_index()
    st.dataframe(pred.style.format({'Total':'{:,.0f}','Canjearan':'{:,.0f}','Nuevos (P>20%)':'{:,.0f}',
        'Revenue esp':'${:,.0f}','Pts por vencer':'{:,.0f}'}),use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="🎯 Motor Decision":
    st.title("🎯 Motor de Decision — 5 criterios de priorizacion")
    st.markdown("""
> No todos los clientes se priorizan igual. Segun el objetivo de negocio, la lista de prioridad cambia.
> El motor combina: P(canje), CLV, uplift causal, estado funnel, y riesgo de fuga.
    """)
    criterio=st.selectbox("Seleccionar criterio:",
        ["📊 Probabilidad de canje (P>50%)","💰 Valor esperado (Revenue × P)",
         "⚡ Uplift causal (quien cambia con campana)","🚨 Riesgo de fuga (urgentes)",
         "🆕 Potencial de activacion (primer canje)"])
    if criterio.startswith("📊"):
        st.markdown("> **Top clientes con mayor probabilidad** de canjear. La campana puede acelerar el timing.")
        top=df.nlargest(20,'prob')[['cust_id','tier','funnel_state_at_t0','prob','cluster_name','prioridad']].copy()
        top['P(canje)']=top.prob.apply(lambda x: f"{x:.1%}")
        st.dataframe(top.drop(columns=['prob']),use_container_width=True)
    elif criterio.startswith("💰"):
        st.markdown("> **Top clientes por valor esperado** = P(canje) × revenue mensual estimado.")
        if 'revenue_esperado_1m' in df.columns:
            top=df.nlargest(20,'revenue_esperado_1m')[['cust_id','tier','funnel_state_at_t0','prob','revenue_esperado_1m','prioridad']].copy()
            top['Revenue esp']=top.revenue_esperado_1m.apply(lambda x: fmt(x))
            st.dataframe(top.drop(columns=['revenue_esperado_1m']),use_container_width=True)
    elif criterio.startswith("⚡"):
        st.markdown("> **Clientes donde la campana CAMBIA el resultado.** Alto uplift = contactar los mueve a canjear.")
        if 'uplift_x' in df.columns:
            top=df.nlargest(20,'uplift_x')[['cust_id','tier','funnel_state_at_t0','prob','uplift_x','prioridad']].copy()
            top['Uplift']=top.uplift_x.apply(lambda x: fmt(x))
            st.dataframe(top.drop(columns=['uplift_x']),use_container_width=True)
            st.info("A diferencia de P(canje), el uplift mide el **efecto causal**. Alto uplift = no canjearia solo, pero si lo contactamos, si.")
    elif criterio.startswith("🚨"):
        st.markdown("> **Clientes en fuga** que dejaron de canjear. Priorizar por CLV para maximizar recuperacion.")
        fuga=df[df.funnel_state_at_t0=='Fuga'].nlargest(20,'clv_estimado')[['cust_id','tier','prob','clv_estimado','days_since_last_redeem','prioridad']].copy()
        if len(fuga)>0:
            fuga['CLV']=fuga.clv_estimado.apply(lambda x: fmt(x))
            fuga['Dias sin canje']=fuga.days_since_last_redeem.astype(int)
            st.dataframe(fuga.drop(columns=['clv_estimado','days_since_last_redeem']),use_container_width=True)
            st.warning("Cada dia sin reactivar sube el costo de recuperacion.")
        else: st.info("No hay clientes en Fuga con los filtros actuales.")
    elif criterio.startswith("🆕"):
        st.markdown("> **Clientes que nunca canjearon** pero tienen puntos y probabilidad. Primer canje = activacion.")
        if 'has_redeemed_before_t0' in df.columns:
            nuevos=df[(df.has_redeemed_before_t0==False)&(df.stock_points_at_t0>=1000)].nlargest(20,'prob')
            nuevos=nuevos[['cust_id','tier','funnel_state_at_t0','prob','stock_points_at_t0','prioridad']].copy()
            nuevos['P(canje)']=nuevos.prob.apply(lambda x: f"{x:.1%}")
            nuevos['Puntos']=nuevos.stock_points_at_t0.apply(lambda x: f"{x:,.0f}")
            st.dataframe(nuevos.drop(columns=['prob','stock_points_at_t0']),use_container_width=True)
            st.success("El **primer canje** es el momento mas critico. Despues de canjear una vez, la probabilidad de recurrencia sube significativamente.")

# ══════════════════════════════════════════════════════════════
elif V=="🎚️ Simulador":
    st.title("🎚️ Simulador de Impacto de Campana")
    st.markdown("""
> **Simula el impacto** de una campana de activacion/reactivacion.
>
> **Supuestos:**
> - Revenue por activacion = gasto mensual promedio de un canjeador
> - Reactivacion de fuga = 70% del revenue de un canjeador nuevo (menor engagement)
> - Participante → canje = 50% del revenue (aun no tiene habito)
> - Ganancia neta = Revenue incremental - Costo total de la campana
    """)
    c1,c2=st.columns(2)
    pos_n=len(df[df.funnel_state_at_t0=='Posibilidad Canje'])
    fuga_n=len(df[df.funnel_state_at_t0=='Fuga'])
    dorm_n=len(df[df.funnel_state_at_t0=='Participante'])
    avg_rev=df[df.y_target==1].revenue_post_12m.mean()/12 if (df.y_target==1).any() else 0

    with c1:
        st.markdown(f"**Universo:** Posibilidad Canje: {pos_n:,} | Fuga: {fuga_n:,} | Participantes: {dorm_n:,}")
        pct_act=st.slider("% Posibilidad Canje que activamos",0,50,15,5)
        pct_react=st.slider("% Fuga que reactivamos",0,30,10,5)
        pct_part=st.slider("% Participantes que empujan a canjear",0,20,5,5)
        costo_contacto=st.number_input("Costo por contacto (CLP)",value=500,step=100)

    n_act=int(pos_n*pct_act/100); n_react=int(fuga_n*pct_react/100); n_part=int(dorm_n*pct_part/100)
    total_contactar=n_act+n_react+n_part
    rev_act=n_act*avg_rev
    rev_react=n_react*avg_rev*0.7
    rev_part=n_part*avg_rev*0.5
    total_rev=rev_act+rev_react+rev_part
    costo_total=total_contactar*costo_contacto
    ganancia_neta=total_rev-costo_total
    roi=ganancia_neta/costo_total*100 if costo_total>0 else 0

    with c2:
        st.metric("Clientes a contactar",f"{total_contactar:,}")
        st.metric("Revenue incremental",fmt(total_rev))
        st.metric("Costo campana",fmt(costo_total))
        st.metric("**Ganancia neta**",fmt(ganancia_neta),delta=f"ROI {roi:,.0f}%")

    # Desglose
    desg = pd.DataFrame([
        {'Segmento':'Posibilidad Canje','N contactar':n_act,'Tasa conversion':f"{pct_act}%",'Revenue':rev_act,'Factor':'100%'},
        {'Segmento':'Fuga','N contactar':n_react,'Tasa conversion':f"{pct_react}%",'Revenue':rev_react,'Factor':'70%'},
        {'Segmento':'Participantes','N contactar':n_part,'Tasa conversion':f"{pct_part}%",'Revenue':rev_part,'Factor':'50%'},
    ])
    st.dataframe(desg.style.format({'N contactar':'{:,.0f}','Revenue':'${:,.0f}'}),use_container_width=True)
    st.caption(f"Revenue mensual promedio por canjeador: {fmt(avg_rev)}")

    if roi>0: st.success(f"La campana genera **{fmt(ganancia_neta)}** de ganancia neta con ROI de **{roi:,.0f}%**.")
    else: st.error("El ROI es negativo. Ajusta los parametros o reduce el costo por contacto.")

# ══════════════════════════════════════════════════════════════
elif V=="👤 Ficha Cliente":
    st.title("👤 Ficha de Cliente")
    st.markdown("> Busca un cliente para ver su perfil completo, prediccion, y recomendacion.")
    ids=sorted(A.cust_id.unique())[:200]
    cid=st.selectbox("Seleccionar cliente",ids)
    c=A[A.cust_id==cid]
    if len(c)==0: st.error("No encontrado")
    else:
        r=c.iloc[0]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Categoria",str(r.get('tier','-'))); c2.metric("Funnel",str(r.get('funnel_state_at_t0','-')))
        c3.metric("Cluster",str(r.get('cluster_name','-'))); c4.metric("Prioridad",str(r.get('prioridad','-')))
        c5,c6,c7,c8=st.columns(4)
        c5.metric("P(canje 1m)",f"{r.prob:.1%}")
        c6.metric("CLV",fmt(r.clv_estimado) if 'clv_estimado' in r.index else "-")
        c7.metric("Gasto 12m",fmt(r.monetary_total) if 'monetary_total' in r.index else "-")
        c8.metric("Puntos",f"{r.stock_points_at_t0:,.0f}" if 'stock_points_at_t0' in r.index else "-")
        _t=M.get('threshold',0.5)
        tipo='Canjeador Nuevo predicho' if r.prob>_t*0.5 and r.get('has_redeemed_before_t0',False)==False else 'Recurrente predicho' if r.prob>_t else 'No canjeara'
        st.info(f"**Prediccion:** {tipo} | **RFM:** {r.get('rfm_segment','?')} | **Breakage:** {r.get('breakage',0)*100:.0f}%")

# ══════════════════════════════════════════════════════════════
elif V=="📋 Aperturas":
    st.title("📋 Aperturas & Cross-tabs")
    st.markdown("> Cruces entre dimensiones para identificar segmentos de oportunidad.")
    tab1,tab2,tab3,tab4=st.tabs(["Quintil × Categoria","Cluster × Funnel","RFM × Categoria","Prioridad × Quintil"])
    with tab1:
        if 'quintil_label' in df.columns:
            ct=df.groupby(['quintil_label','tier'],observed=True).apply(lambda x: x.y_target.mean()*100).unstack(fill_value=0)
            ct = ct.reindex(columns=[c2 for c2 in TIER_ORDER if c2 in ct.columns])
            fig=px.imshow(ct.values,x=[str(c2) for c2 in ct.columns],y=ct.index.tolist(),
                          color_continuous_scale='RdYlGn',aspect='auto',
                          title="% canje 1m: Quintil × Categoria",labels=dict(color="%"))
            st.plotly_chart(fig,use_container_width=True)
    with tab2:
        if 'cluster_name' in df.columns:
            ct2=pd.crosstab(df.cluster_name,df.funnel_state_at_t0)
            ct2 = ct2.reindex(columns=[c2 for c2 in FUNNEL_ORDER if c2 in ct2.columns])
            fig2=px.imshow(ct2.values,x=[str(c2) for c2 in ct2.columns],y=ct2.index.tolist(),
                           color_continuous_scale='Blues',aspect='auto',title="N clientes: Cluster × Funnel")
            st.plotly_chart(fig2,use_container_width=True)
    with tab3:
        if 'rfm_segment' in df.columns:
            ct3=df.groupby(['rfm_segment','tier'],observed=True).apply(lambda x: x.y_target.mean()*100).unstack(fill_value=0)
            ct3 = ct3.reindex(columns=[c2 for c2 in TIER_ORDER if c2 in ct3.columns])
            fig3=px.imshow(ct3.values,x=[str(c2) for c2 in ct3.columns],y=ct3.index.tolist(),
                           color_continuous_scale='RdYlGn',aspect='auto',title="% canje: RFM × Categoria")
            st.plotly_chart(fig3,use_container_width=True)
    with tab4:
        if 'prioridad' in df.columns and 'quintil_label' in df.columns:
            ct4=pd.crosstab(df.quintil_label,df.prioridad,normalize='index')*100
            cols=[c2 for c2 in PRIORIDAD_ORDER if c2 in ct4.columns]
            if cols:
                fig4=px.bar(ct4[cols].reset_index(),x='quintil_label',y=cols,barmode='stack',
                            title="Prioridad por quintil",category_orders=CAT_ORDERS)
                st.plotly_chart(fig4,use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="💾 Exportar":
    st.title("💾 Exportar listas de clientes")
    st.markdown("> Descarga listas segmentadas para campanas.")
    for gc2,gn in [('prioridad','Prioridad'),('cluster_name','Cluster'),('rfm_segment','RFM')]:
        if gc2 not in df.columns: continue
        st.subheader(f"Por {gn}")
        for v2 in sorted(df[gc2].dropna().unique()):
            s=df[df[gc2]==v2]
            cols=[c2 for c2 in ['cust_id','tier','funnel_state_at_t0','cluster_name','rfm_segment',
                                'prob','clv_estimado','prioridad','revenue_esperado_1m'] if c2 in s.columns]
            csv=s[cols].to_csv(index=False)
            st.download_button(f"📥 {v2} ({len(s):,})",csv,f"{gn}_{str(v2).lower().replace(' ','_')}.csv")
