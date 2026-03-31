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
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Loyalty Intelligence",page_icon="🎯",layout="wide")

@st.cache_data(ttl=3600,show_spinner="Cargando datos...")
def load():
    df=pd.read_csv('data_scored.csv.gz',compression='gzip')
    tmp=pd.read_csv('data_temporal.csv')
    fevo=pd.read_csv('data_funnel_evo.csv')
    met=json.load(open('data_metrics.json'))
    for c in df.columns:
        if c not in ['cust_id','tier','gender','city','dominant_retailer','funnel_state_at_t0','status','cluster_name','prioridad','canal','timing','objetivo','accion','tipo_cliente','quintil_label','target_label','target_label_12m','rfm_segment','t0','tipo_predicho','tipo_canje']:
            try: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
            except: pass
    if 'quintil_label' not in df.columns and 'quintil_gasto' in df.columns:
        df['quintil_label']=df['quintil_gasto'].map({1:'Q1',2:'Q2',3:'Q3',4:'Q4',5:'Q5'})
    if 'breakage' not in df.columns: df['breakage']=(1-df['redeem_rate'].clip(0,1))
    if 'clv_estimado' not in df.columns: df['clv_estimado']=df['revenue_post_12m']*1.5
    # Use 1-month target
    if 'y_1m' in df.columns: df['y_target']=df['y_1m']
    else: df['y_target']=(df['y']>0).astype(int)
    if 'p_canje_1m' in df.columns: df['prob']=df['p_canje_1m']
    elif 'propensity_score' in df.columns: df['prob']=df['propensity_score']
    else: df['prob']=0
    return df,tmp,fevo,met

A,T,FE,M=load()

# Sidebar
st.sidebar.title("🎯 Loyalty Intelligence")
st.sidebar.caption(f"CMR Puntos | {M['n_clientes']:,} clientes | Modelo: {M.get('horizonte','12 meses')}")
V=st.sidebar.radio("",["🏠 Resumen","💰 CLV & Revenue","📊 RFM","🔄 Funnel","🧩 Segmentos","📈 Modelo","⚡ Incrementalidad","🎮 Puntos & Breakage","🔮 Prediccion Mes","🎯 Motor Decision","🎚️ Simulador","👤 Ficha Cliente","📋 Aperturas","💾 Exportar"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros**")
def safe_opts(col): return sorted([x for x in A[col].dropna().unique() if str(x) not in ['','Desconocido','UNKNOWN','nan']]) if col in A.columns else []
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

# ══════════════════════════════════════════════════════════════
if V=="🏠 Resumen":
    st.title("🏠 Resumen Ejecutivo")
    st.markdown(f"*Modelo binario — predice P(canje proximo mes) | AUC: {M['auc']:.2f}*")
    k1,k2,k3,k4,k5,k6=st.columns(6)
    k1.metric("Clientes",f"{df.cust_id.nunique():,}")
    k2.metric("Tasa canje 1m",f"{df.y_target.mean()*100:.1f}%")
    k3.metric("AUC",f"{M['auc']:.3f}")
    k4.metric("Canjearan (pred)",f"{(df.prob>0.5).sum():,}")
    k5.metric("Gasto Prom",fmt(df.monetary_total.mean()))
    k6.metric("% Urgente",f"{(df.prioridad=='Urgente').mean()*100:.1f}%")
    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("💡 Oportunidad: Activar primeros canjes")
        nunca=df[(df.has_redeemed_before_t0==False)&(df.stock_points_at_t0>=1000)]
        prob_alta=nunca[nunca.prob>0.2]
        st.metric("Clientes que nunca canjearon + tienen puntos",f"{len(nunca):,}")
        st.metric("Con P(canje)>20% — alta probabilidad",f"{len(prob_alta):,}")
        rev=len(prob_alta)*df[df.y_target==1].revenue_post_12m.mean()/12 if (df.y_target==1).any() else 0
        st.info(f"**{len(prob_alta):,}** clientes listos para su primer canje. Revenue potencial proximo mes: **{fmt(rev)}**")
    with c2:
        st.subheader("⚠️ Fuga & Urgencia")
        fuga=df[df.funnel_state_at_t0=='Fuga']
        urg=df[df.prioridad=='Urgente']
        st.metric("En Fuga",f"{len(fuga):,}")
        st.metric("Prioridad Urgente",f"{len(urg):,}")
        st.warning(f"**{len(fuga):,}** clientes en fuga. Requieren campaña inmediata de reactivacion.")
    st.subheader("Tasa canje por categoria")
    td=df.groupby('tier').apply(lambda x: pd.Series({'Canjea 1m':x.y_target.mean()*100,'P(canje) promedio':x.prob.mean()*100})).reset_index()
    fig=px.bar(td,x='tier',y='Canjea 1m',color='P(canje) promedio',color_continuous_scale='RdYlGn',title="% que canjea en 1 mes por categoria")
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
    st.markdown("*Customer Lifetime Value: cuanto vale cada tipo de cliente*")
    clv_c=df[df.y_target==1].clv_estimado.mean() if (df.y_target==1).any() else 0
    clv_n=df[df.y_target==0].clv_estimado.mean() if (df.y_target==0).any() else 0
    ratio=clv_c/clv_n if clv_n>0 else 0
    c1,c2,c3=st.columns(3)
    c1.metric("CLV Canjeador",fmt(clv_c)); c2.metric("CLV No canjeador",fmt(clv_n)); c3.metric("Multiplicador",f"{ratio:.1f}x")
    st.success(f"Cada cliente que canjea genera **{ratio:.1f}x mas valor**. Activar 100 clientes = {fmt(100*(clv_c-clv_n))} en CLV adicional.")
    st.subheader("CLV por categoria")
    ct=df.groupby('tier').apply(lambda x: pd.Series({'CLV Canjea':x[x.y_target==1].clv_estimado.mean() if (x.y_target==1).any() else 0,'CLV No canjea':x[x.y_target==0].clv_estimado.mean() if (x.y_target==0).any() else 0})).reset_index()
    fig=px.bar(ct.melt(id_vars='tier'),x='tier',y='value',color='variable',barmode='group',title="CLV por categoria y comportamiento",labels={'value':'CLV ($)','tier':'Categoria'})
    st.plotly_chart(fig,use_container_width=True)
    if 'revenue_esperado_1m' in df.columns:
        st.subheader("Revenue esperado proximo mes")
        rev_tier=df.groupby('tier').revenue_esperado_1m.sum().reset_index()
        fig2=px.bar(rev_tier,x='tier',y='revenue_esperado_1m',title="Revenue esperado por categoria",labels={'revenue_esperado_1m':'Revenue ($)'})
        st.plotly_chart(fig2,use_container_width=True)
        st.metric("Revenue total esperado proximo mes",fmt(df.revenue_esperado_1m.sum()))

# ══════════════════════════════════════════════════════════════
elif V=="📊 RFM":
    st.title("📊 Analisis RFM")
    st.markdown("""*Segmentacion clasica de CRM: **R**ecency (dias desde ultima compra) × **F**requency (compras) × **M**onetary (gasto)*

- **Champions** = recientes + frecuentes + alto gasto → retener y premiar
- **Loyal** = buenos en las 3 dimensiones → mantener engagement
- **En Riesgo** = fueron buenos pero se estan yendo → campaña urgente
- **Perdidos** = inactivos, baja frecuencia, bajo gasto → reactivar o soltar
- **Nuevos** = recientes pero poca frecuencia → desarrollar habito
    """)
    if 'rfm_segment' in df.columns:
        rfm=df.groupby('rfm_segment').apply(lambda x: pd.Series({'Clientes':len(x),'%':len(x)/len(df)*100,'Tasa canje':x.y_target.mean()*100,'Recency (dias)':x.recency_days.mean(),'Frecuencia':x.frequency_total.mean(),'Gasto prom':x.monetary_total.mean(),'CLV':x.clv_estimado.mean()})).reset_index().sort_values('CLV',ascending=False)
        st.dataframe(rfm.style.format({'Clientes':'{:,.0f}','%':'{:.1f}%','Tasa canje':'{:.1f}%','Recency (dias)':'{:.0f}','Frecuencia':'{:.1f}','Gasto prom':'${:,.0f}','CLV':'${:,.0f}'}),use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            fig=px.pie(rfm,names='rfm_segment',values='Clientes',hole=.4,title="Distribucion RFM")
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            samp=df.sample(min(2000,len(df)),random_state=42)
            fig2=px.scatter(samp,x='recency_days',y='frequency_total',size=np.clip(samp.monetary_total,0,samp.monetary_total.quantile(0.95)),color='rfm_segment',title="Recency × Frecuencia (tamaño=gasto)",opacity=0.5)
            fig2.update_layout(xaxis_title="Dias desde ultima compra (menor=mejor)",yaxis_title="N compras 12m")
            st.plotly_chart(fig2,use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="🔄 Funnel":
    st.title("🔄 Funnel de Canje")
    st.markdown("""*El viaje del cliente en el programa:*

**Inscrito** → **Participante** (compra) → **Posibilidad Canje** (tiene ≥1000 pts) → **Canjeador** (1 canje) → **Recurrente** (2+ canjes)
↘ **Fuga** (dejo de canjear >12 meses)
    """)
    states=['Inscrito','Participante','Posibilidad Canje','Canjeador','Recurrente','Fuga']
    if 't0_label' in FE.columns:
        fig=px.area(FE,x='t0_label',y='n',color='funnel_state_at_t0',category_orders={'funnel_state_at_t0':states},color_discrete_sequence=px.colors.qualitative.Set2,title="Evolucion del funnel")
        fig.update_layout(yaxis_title="Clientes",xaxis_title="Mes",legend_title="Estado")
        st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        fd=df.funnel_state_at_t0.value_counts(); fig2=px.pie(values=fd.values,names=fd.index,hole=.4,title="Hoy")
        st.plotly_chart(fig2,use_container_width=True)
    with c2:
        fr=df.groupby('funnel_state_at_t0').apply(lambda x: x.y_target.mean()*100).reset_index(name='Tasa 1m').sort_values('Tasa 1m',ascending=False)
        fig3=px.bar(fr,x='funnel_state_at_t0',y='Tasa 1m',color='Tasa 1m',color_continuous_scale='RdYlGn',title="P(canje proximo mes) por estado")
        fig3.update_layout(yaxis_title="% canjea",xaxis_title="Estado")
        st.plotly_chart(fig3,use_container_width=True)
    big=df.funnel_state_at_t0.value_counts()
    st.info(f"🔍 **Cuello de botella:** {big.idxmax()} ({big.max():,} clientes, {big.max()/len(df)*100:.0f}%). Oportunidad: mover a estos clientes al siguiente estado.")

# ══════════════════════════════════════════════════════════════
elif V=="🧩 Segmentos":
    st.title("🧩 Segmentos (K-Means)")
    if 'cluster_name' in df.columns:
        cl=df.groupby('cluster_name').apply(lambda x: pd.Series({'N':len(x),'Tasa 1m':x.y_target.mean()*100,'P(canje)':x.prob.mean(),'CLV':x.clv_estimado.mean(),'Gasto prom':x.monetary_total.mean()})).reset_index().sort_values('CLV',ascending=False)
        st.dataframe(cl.style.format({'N':'{:,.0f}','Tasa 1m':'{:.1f}%','P(canje)':'{:.3f}','CLV':'${:,.0f}','Gasto prom':'${:,.0f}'}),use_container_width=True)
        acc={'Heavy Users':'🏆 Upgrade tier, experiencia exclusiva, early access','Exploradores':'🔍 Ofertas multi-retailer, cross-sell entre tiendas','Cazadores de Canje':'🎁 Descuento 20% puntos, catalogo personalizado','Digitales':'📱 Push notifications, canje express en app','Dormidos':'⏰ Puntos x2, oferta de bienvenida, email personalizado','En Riesgo':'🚨 Alerta puntos por vencer, llamada directa'}
        st.subheader("Accion recomendada")
        for cn in sorted(df.cluster_name.dropna().unique()):
            n=len(df[df.cluster_name==cn]); st.markdown(f"**{cn}** ({n:,}): {acc.get(cn,'Personalizar')}")

# ══════════════════════════════════════════════════════════════
elif V=="📈 Modelo":
    st.title("📈 Modelo Predictivo")
    st.markdown(f"*XGBoost binario — P(canje en proximo mes) | Horizonte: {M.get('horizonte','1 mes')}*")
    try:
        from sklearn.metrics import roc_auc_score,roc_curve
        yb=df.y_target.astype(int); pp=df.prob
        auc=roc_auc_score(yb,pp)
        c1,c2,c3=st.columns(3); c1.metric("AUC",f"{auc:.4f}"); c2.metric("Tasa base",f"{yb.mean()*100:.1f}%"); c3.metric("Clientes",f"{len(df):,}")
        fpr,tpr,_=roc_curve(yb,pp)
        fig=go.Figure(); fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f'AUC={auc:.3f}',line=dict(width=3,color='steelblue'))); fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(dash='dash',color='gray'),name='Aleatorio'))
        fig.update_layout(title='Curva ROC',xaxis_title='Tasa falsos positivos',yaxis_title='Tasa verdaderos positivos')
        st.plotly_chart(fig,use_container_width=True)
        st.subheader("Lift — Eficiencia de targeting")
        base=yb.mean(); dt=df.copy(); dt['dec']=pd.qcut(dt.prob.rank(method='first'),10,labels=range(1,11))
        ld=[]; cum=0; tot=yb.sum()
        for d in range(10,0,-1):
            s=dt[dt.dec==d]; r=s.y_target.sum(); cum+=r
            ld.append({'Decil':d,'Tasa':r/len(s)*100,'Lift':r/len(s)/base if base>0 else 0,'Captura':cum/tot*100 if tot>0 else 0})
        ldf=pd.DataFrame(ld)
        fig2=make_subplots(specs=[[{"secondary_y":True}]])
        fig2.add_trace(go.Bar(x=ldf.Decil,y=ldf.Lift,name='Lift vs base',marker_color='steelblue'),secondary_y=False)
        fig2.add_trace(go.Scatter(x=ldf.Decil,y=ldf.Captura,name='% capturado',mode='lines+markers',marker_color='red'),secondary_y=True)
        fig2.update_layout(title='Si contactamos al top X%, cuantos canjeadores capturamos?',xaxis=dict(dtick=1))
        fig2.update_yaxes(title_text='Lift (veces sobre base)',secondary_y=False)
        fig2.update_yaxes(title_text='% canjeadores capturados',secondary_y=True)
        st.plotly_chart(fig2,use_container_width=True)
        t30=ldf[ldf.Decil>=8].Captura.max() if len(ldf)>0 else 0
        st.success(f"Contactando al **top 30%** capturamos el **{t30:.0f}%** de los canjeadores del proximo mes.")
    except Exception as e: st.warning(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
elif V=="⚡ Incrementalidad":
    st.title("⚡ Incrementalidad")
    st.markdown("""*Cuanto gasto **adicional** genera el canje? Comparamos el metodo actual vs el causal:*
- **GitLab (actual):** compara canjeadores vs potenciales directamente → sobreestima porque los canjeadores ya eran mas activos
- **PSM (causal):** empareja clientes similares → mide el efecto real del canje
    """)
    gl=pd.DataFrame([{'Retail':'FALABELLA','GitLab':28.1,'PSM':21.2},{'Retail':'SODIMAC','GitLab':25.0,'PSM':3.9},{'Retail':'TOTTUS','GitLab':37.0,'PSM':23.5},{'Retail':'FCOM','GitLab':19.3,'PSM':28.9}])
    c1,c2,c3=st.columns(3); c1.metric("Lift GitLab","+35.3%"); c2.metric("Lift PSM (real)","+16.2%"); c3.metric("Sobreestimacion","2.2x")
    fig=go.Figure(); fig.add_trace(go.Bar(x=gl.Retail,y=gl.GitLab,name='GitLab (reportado)',marker_color='gray')); fig.add_trace(go.Bar(x=gl.Retail,y=gl.PSM,name='PSM (causal real)',marker_color='steelblue'))
    fig.update_layout(barmode='group',title='Lift por retail: lo que se reporta vs la realidad',yaxis_title='Lift gasto %',xaxis_title='Retail')
    st.plotly_chart(fig,use_container_width=True)
    st.info("De cada **$100** que GitLab reporta como incrementalidad, solo **$46** son realmente causados por el canje. El resto es autoseleccion.")
    if 'uplift_x' in df.columns:
        p1,p99=df.uplift_x.quantile(0.02),df.uplift_x.quantile(0.98)
        dc=df[(df.uplift_x>=p1)&(df.uplift_x<=p99)]
        fig2=px.histogram(dc,x='uplift_x',nbins=50,color='prioridad',color_discrete_map={'Alta':'green','Media':'orange','Baja':'gray','No contactar':'red','Urgente':'darkred'},title="Efecto individual: a quien le sirve que lo contactemos?")
        fig2.update_layout(xaxis_title="Uplift ($) — positivo = vale la pena contactar",yaxis_title="Clientes")
        st.plotly_chart(fig2,use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="🎮 Puntos & Breakage":
    st.title("🎮 Puntos, Breakage & Gamificacion")
    st.markdown("""*El breakage es el % de puntos que se acumulan pero **nunca se canjean**.
Cada punto no canjeado es revenue potencial perdido. Reducir el breakage = activar gasto.*""")
    brk=df.breakage.mean()*100 if 'breakage' in df.columns else 75
    pts_stock=df.stock_points_at_t0.sum()
    pts_vencer=df.exp_points_current_at_t0.sum() if 'exp_points_current_at_t0' in df.columns else 0
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Breakage promedio",f"{brk:.0f}%")
    c2.metric("Stock total puntos",f"{pts_stock:,.0f}")
    c3.metric("Puntos por vencer",f"{pts_vencer:,.0f}")
    c4.metric("Valor CLP (6.5/pto)",fmt(pts_vencer*6.5))
    st.subheader("Breakage por categoria")
    if 'breakage' in df.columns:
        bk=df.groupby('tier').apply(lambda x: pd.Series({'Breakage':x.breakage.mean()*100,'Stock promedio':x.stock_points_at_t0.mean(),'Puntos x vencer':x.exp_points_current_at_t0.mean() if 'exp_points_current_at_t0' in x.columns else 0})).reset_index()
        fig=px.bar(bk,x='tier',y='Breakage',color='Breakage',color_continuous_scale='RdYlGn_r',title="% puntos que nunca se canjean — menor es mejor")
        fig.update_layout(yaxis_title="Breakage %",xaxis_title="Categoria")
        st.plotly_chart(fig,use_container_width=True)
    st.subheader("💡 Simulador: si reducimos el breakage...")
    target_brk=st.slider("Nuevo breakage objetivo:",min_value=max(int(brk-30),10),max_value=int(brk),value=int(brk-10),step=5)
    delta=(brk-target_brk)/100
    pts_add=pts_stock*delta; rev_add=pts_add*6.5
    st.metric(f"Puntos adicionales canjeados si breakage baja de {brk:.0f}% a {target_brk}%",f"{pts_add:,.0f}")
    st.metric("Revenue adicional estimado",fmt(rev_add))
    st.success(f"Reducir el breakage **{brk:.0f}% → {target_brk}%** libera **{pts_add:,.0f} puntos** = **{fmt(rev_add)}** en gasto adicional.")

# ══════════════════════════════════════════════════════════════
elif V=="🔮 Prediccion Mes":
    st.title("🔮 Que esperamos el proximo mes")
    st.markdown("*Basado en el modelo binario de 1 mes, estas son las predicciones:*")
    n_canjean=int((df.prob>0.5).sum())
    n_nuevos=int(((df.prob>0.2)&(df.has_redeemed_before_t0==False)).sum()) if 'has_redeemed_before_t0' in df.columns else 0
    n_recur=int(((df.prob>0.5)&(df.has_redeemed_before_t0==True)).sum()) if 'has_redeemed_before_t0' in df.columns else 0
    rev_esp=df.revenue_esperado_1m.sum() if 'revenue_esperado_1m' in df.columns else 0
    pts_esp=df.puntos_esperados_canje.sum() if 'puntos_esperados_canje' in df.columns else 0
    pts_vencer=df.exp_points_current_at_t0.sum() if 'exp_points_current_at_t0' in df.columns else 0
    c1,c2,c3=st.columns(3)
    c1.metric("🎯 Canjearan (P>50%)",f"{n_canjean:,}")
    c2.metric("🆕 Canjeadores nuevos (P>20%)",f"{n_nuevos:,}")
    c3.metric("🔁 Recurrentes (P>50%)",f"{n_recur:,}")
    c4,c5,c6=st.columns(3)
    c4.metric("💰 Revenue esperado",fmt(rev_esp))
    c5.metric("🎁 Puntos a canjear",f"{pts_esp:,.0f}")
    c6.metric("⏰ Puntos por vencer",f"{pts_vencer:,.0f}")
    st.markdown("""
    > **Revenue esperado** = Σ P(canje) × gasto mensual de cada cliente
    > **Puntos a canjear** = Σ P(canje) × promedio de puntos por canje
    > **Canjeador nuevo** = nunca canjeo antes + P(canje) > 20%
    """)
    st.subheader("Apertura por categoria")
    pred=df.groupby('tier').apply(lambda x: pd.Series({
        'Canjearan (P>50%)':(x.prob>0.5).sum(),
        'Nuevos (P>20%)':((x.prob>0.2)&(x.has_redeemed_before_t0==False)).sum() if 'has_redeemed_before_t0' in x.columns else 0,
        'Revenue esp':x.revenue_esperado_1m.sum() if 'revenue_esperado_1m' in x.columns else 0,
        'Pts por vencer':x.exp_points_current_at_t0.sum() if 'exp_points_current_at_t0' in x.columns else 0,
    })).reset_index()
    st.dataframe(pred.style.format({'Canjearan (P>50%)':'{:,.0f}','Nuevos (P>20%)':'{:,.0f}','Revenue esp':'${:,.0f}','Pts por vencer':'{:,.0f}'}),use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="🎯 Motor Decision":
    st.title("🎯 Motor de Decision — 5 criterios de priorizacion")
    st.markdown("*No todos los clientes se priorizan igual. Segun el objetivo, la lista cambia:*")
    criterio=st.selectbox("Seleccionar criterio de priorizacion:",["📊 Probabilidad de canje (P>50%)","💰 Valor esperado (Revenue × P)","⚡ Uplift causal (quién cambia con campana)","🚨 Riesgo de fuga (urgentes)","🆕 Potencial de activacion (primer canje)"])
    if criterio.startswith("📊"):
        st.markdown("**Top clientes con mayor probabilidad de canjear el proximo mes.**")
        top=df.nlargest(20,'prob')[['cust_id','tier','funnel_state_at_t0','prob','cluster_name','prioridad']].copy()
        top['P(canje)']=top.prob.apply(lambda x: f"{x:.1%}")
        st.dataframe(top.drop(columns=['prob']),use_container_width=True)
        st.info("Estos clientes van a canjear casi seguro. La campaña puede acelerar el timing pero no cambia la decision.")
    elif criterio.startswith("💰"):
        st.markdown("**Top clientes con mayor valor esperado = P(canje) × revenue estimado.**")
        if 'revenue_esperado_1m' in df.columns:
            top=df.nlargest(20,'revenue_esperado_1m')[['cust_id','tier','funnel_state_at_t0','prob','revenue_esperado_1m','prioridad']].copy()
            top['Revenue esp']=top.revenue_esperado_1m.apply(lambda x: fmt(x))
            st.dataframe(top.drop(columns=['revenue_esperado_1m']),use_container_width=True)
    elif criterio.startswith("⚡"):
        st.markdown("**Clientes donde la campaña CAMBIA el resultado. Alto uplift = contactar los mueve.**")
        if 'uplift_x' in df.columns:
            top=df.nlargest(20,'uplift_x')[['cust_id','tier','funnel_state_at_t0','prob','uplift_x','prioridad']].copy()
            top['Uplift']=top.uplift_x.apply(lambda x: fmt(x))
            st.dataframe(top.drop(columns=['uplift_x']),use_container_width=True)
            st.info("A diferencia de P(canje), el uplift mide el **efecto causal** de contactar. Alto uplift = el cliente no canjearia solo, pero si lo contactamos, si.")
    elif criterio.startswith("🚨"):
        st.markdown("**Clientes que dejaron de canjear y necesitan reactivacion inmediata.**")
        fuga=df[df.funnel_state_at_t0=='Fuga'].nlargest(20,'clv_estimado')[['cust_id','tier','prob','clv_estimado','days_since_last_redeem','prioridad']].copy()
        if len(fuga)>0:
            fuga['CLV']=fuga.clv_estimado.apply(lambda x: fmt(x))
            fuga['Dias sin canje']=fuga.days_since_last_redeem.astype(int)
            st.dataframe(fuga.drop(columns=['clv_estimado','days_since_last_redeem']),use_container_width=True)
            st.warning("Cada dia que pasa sin reactivar a estos clientes, el costo de reactivacion sube.")
        else: st.info("No hay clientes en Fuga con los filtros actuales.")
    elif criterio.startswith("🆕"):
        st.markdown("**Clientes que nunca canjearon pero tienen puntos y probabilidad. Primer canje = activacion.**")
        if 'has_redeemed_before_t0' in df.columns:
            nuevos=df[(df.has_redeemed_before_t0==False)&(df.stock_points_at_t0>=1000)].nlargest(20,'prob')[['cust_id','tier','funnel_state_at_t0','prob','stock_points_at_t0','prioridad']].copy()
            nuevos['P(canje)']=nuevos.prob.apply(lambda x: f"{x:.1%}")
            nuevos['Puntos']=nuevos.stock_points_at_t0.apply(lambda x: f"{x:,.0f}")
            st.dataframe(nuevos.drop(columns=['prob','stock_points_at_t0']),use_container_width=True)
            st.success("El **primer canje** es el momento mas critico. Despues de canjear una vez, la probabilidad de recurrencia sube significativamente.")

# ══════════════════════════════════════════════════════════════
elif V=="🎚️ Simulador":
    st.title("🎚️ Simulador de Impacto")
    st.markdown("*Ajusta los parametros y calcula el impacto en revenue*")
    c1,c2=st.columns(2)
    pos_n=len(df[df.funnel_state_at_t0=='Posibilidad Canje'])
    fuga_n=len(df[df.funnel_state_at_t0=='Fuga'])
    dorm_n=len(df[df.funnel_state_at_t0=='Participante'])
    avg_rev=df[df.y_target==1].revenue_post_12m.mean()/12 if (df.y_target==1).any() else 0
    with c1:
        pct_act=st.slider("% Posibilidad Canje que activamos",0,50,15,5)
        pct_react=st.slider("% Fuga que reactivamos",0,30,10,5)
        pct_part=st.slider("% Participantes que empujan a canjear",0,20,5,5)
        costo_contacto=st.number_input("Costo por contacto (CLP)",value=500,step=100)
    n_act=int(pos_n*pct_act/100); n_react=int(fuga_n*pct_react/100); n_part=int(dorm_n*pct_part/100)
    total_contactar=n_act+n_react+n_part
    rev_act=n_act*avg_rev; rev_react=n_react*avg_rev*0.7; rev_part=n_part*avg_rev*0.5
    total_rev=rev_act+rev_react+rev_part
    costo_total=total_contactar*costo_contacto
    roi=(total_rev-costo_total)/costo_total*100 if costo_total>0 else 0
    with c2:
        st.metric("Clientes a contactar",f"{total_contactar:,}")
        st.metric("Revenue incremental",fmt(total_rev))
        st.metric("Costo campaña",fmt(costo_total))
        st.metric("ROI",f"{roi:,.0f}%")
        st.metric("**Ganancia neta**",fmt(total_rev-costo_total))
    if roi>0: st.success(f"La campaña genera **{fmt(total_rev-costo_total)}** de ganancia neta con ROI de **{roi:,.0f}%**.")
    else: st.error("El ROI es negativo. Ajusta los parametros.")

# ══════════════════════════════════════════════════════════════
elif V=="👤 Ficha Cliente":
    st.title("👤 Ficha de Cliente")
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
        tipo='Canjeador Nuevo predicho' if r.prob>0.3 and r.get('has_redeemed_before_t0',False)==False else 'Recurrente predicho' if r.prob>0.5 else 'No canjeara'
        st.info(f"**Prediccion:** {tipo} | **RFM:** {r.get('rfm_segment','?')} | **Breakage:** {r.get('breakage',0)*100:.0f}%")

# ══════════════════════════════════════════════════════════════
elif V=="📋 Aperturas":
    st.title("📋 Aperturas & Cross-tabs")
    tab1,tab2,tab3,tab4=st.tabs(["Quintil × Categoria","Cluster × Funnel","RFM × Categoria","Prioridad × Quintil"])
    with tab1:
        if 'quintil_label' in df.columns:
            ct=df.groupby(['quintil_label','tier']).apply(lambda x: x.y_target.mean()*100).unstack(fill_value=0)
            fig=px.imshow(ct.values,x=ct.columns.tolist(),y=ct.index.tolist(),color_continuous_scale='RdYlGn',aspect='auto',title="% canje 1m: Quintil × Categoria",labels=dict(color="%"))
            st.plotly_chart(fig,use_container_width=True)
    with tab2:
        if 'cluster_name' in df.columns:
            ct2=pd.crosstab(df.cluster_name,df.funnel_state_at_t0)
            fig2=px.imshow(ct2.values,x=ct2.columns.tolist(),y=ct2.index.tolist(),color_continuous_scale='Blues',aspect='auto',title="N clientes: Cluster × Funnel")
            st.plotly_chart(fig2,use_container_width=True)
    with tab3:
        if 'rfm_segment' in df.columns:
            ct3=df.groupby(['rfm_segment','tier']).apply(lambda x: x.y_target.mean()*100).unstack(fill_value=0)
            fig3=px.imshow(ct3.values,x=ct3.columns.tolist(),y=ct3.index.tolist(),color_continuous_scale='RdYlGn',aspect='auto',title="% canje: RFM × Categoria")
            st.plotly_chart(fig3,use_container_width=True)
    with tab4:
        if 'prioridad' in df.columns and 'quintil_label' in df.columns:
            ct4=pd.crosstab(df.quintil_label,df.prioridad,normalize='index')*100
            cols=[c2 for c2 in ['Alta','Media','Baja','Urgente','No contactar'] if c2 in ct4.columns]
            fig4=px.bar(ct4[cols].reset_index(),x='quintil_label',y=cols,barmode='stack',title="Prioridad por quintil")
            st.plotly_chart(fig4,use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="💾 Exportar":
    st.title("💾 Exportar listas")
    for gc2,gn in [('prioridad','Prioridad'),('cluster_name','Cluster'),('rfm_segment','RFM')]:
        if gc2 not in df.columns: continue
        st.subheader(f"Por {gn}")
        for v2 in sorted(df[gc2].dropna().unique()):
            s=df[df[gc2]==v2]
            cols=[c2 for c2 in ['cust_id','tier','funnel_state_at_t0','cluster_name','rfm_segment','prob','clv_estimado','prioridad','revenue_esperado_1m'] if c2 in s.columns]
            csv=s[cols].to_csv(index=False)
            st.download_button(f"📥 {v2} ({len(s):,})",csv,f"{gn}_{str(v2).lower().replace(' ','_')}.csv")
