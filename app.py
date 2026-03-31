"""🎯 Loyalty Intelligence System — CMR Puntos"""
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
    df=pd.read_csv('data_scored.csv.gz',compression='gzip')
    tmp=pd.read_csv('data_temporal.csv')
    fevo=pd.read_csv('data_funnel_evo.csv')
    met=json.load(open('data_metrics.json'))
    for c in df.columns:
        if c not in ['cust_id','tier','gender','city','dominant_retailer','funnel_state_at_t0','status','cluster_name','prioridad','canal','timing','objetivo','accion','tipo_cliente','quintil_label','target_label','rfm_segment','t0']:
            try: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
            except: pass
    if 'quintil_label' not in df.columns and 'quintil_gasto' in df.columns:
        df['quintil_label']=df['quintil_gasto'].map({1:'Q1',2:'Q2',3:'Q3',4:'Q4',5:'Q5'})
    if 'tipo_cliente' not in df.columns:
        df['tipo_cliente']=np.where(df.y>=1,'Canjeador',np.where(df.stock_points_at_t0>=1000,'Potencial','Acumulador'))
    if 'target_label' not in df.columns:
        df['target_label']=df['y'].map({0:'No canjea',1:'Activacion',2:'Recurrencia'})
    if 'clv_estimado' not in df.columns: df['clv_estimado']=df['revenue_post_12m']*1.5
    if 'breakage' not in df.columns: df['breakage']=1-df['redeem_rate'].clip(0,1)
    return df,tmp,fevo,met

A,T,F,M=load()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🎯 Loyalty Intelligence")
st.sidebar.caption(f"CMR Puntos | {M['n_clientes']:,} clientes reales")
V=st.sidebar.radio("",["🏠 Resumen Ejecutivo","💰 CLV & Revenue","📊 RFM","🔄 Funnel","🧩 Segmentos","📈 Modelo","⚡ Incrementalidad","🎮 Gamificacion","🔮 Prediccion Proximo Mes","🎯 Simulador de Impacto","👤 Ficha Cliente","📋 Aperturas","💾 Exportar"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros**")
tiers=st.sidebar.multiselect("Categoria",sorted([x for x in A.tier.dropna().unique() if x not in ['','Desconocido','UNKNOWN']]),default=sorted([x for x in A.tier.dropna().unique() if x not in ['','Desconocido','UNKNOWN']]))
if 'cluster_name' in A.columns:
    cls=st.sidebar.multiselect("Cluster",sorted([x for x in A.cluster_name.dropna().unique() if x]),default=sorted([x for x in A.cluster_name.dropna().unique() if x]))
else: cls=[]
fns=st.sidebar.multiselect("Funnel",sorted([x for x in A.funnel_state_at_t0.dropna().unique() if x not in ['','Desconocido','UNKNOWN']]),default=sorted([x for x in A.funnel_state_at_t0.dropna().unique() if x not in ['','Desconocido','UNKNOWN']]))
qts=st.sidebar.multiselect("Quintil",sorted([x for x in A.quintil_label.dropna().unique() if x]) if 'quintil_label' in A.columns else [],default=sorted([x for x in A.quintil_label.dropna().unique() if x]) if 'quintil_label' in A.columns else [])
prs=st.sidebar.multiselect("Prioridad",['Alta','Media','Baja','No contactar'],default=['Alta','Media','Baja','No contactar'])

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

# helper
def safe_pct(s): return s.mean()*100 if len(s)>0 else 0
def fmt_clp(v): return f"${v:,.0f}" if abs(v)<1e9 else f"${v/1e6:,.0f}M"

# ══════════════════════════════════════════════════════════════
if V=="🏠 Resumen Ejecutivo":
    st.title("🏠 Resumen Ejecutivo")
    st.markdown("*Vision general del programa y oportunidades de negocio*")
    k1,k2,k3,k4,k5,k6=st.columns(6)
    k1.metric("Clientes",f"{df.cust_id.nunique():,}")
    k2.metric("Tasa Canje",f"{safe_pct(df.y>0):.1f}%")
    k3.metric("AUC Modelo",f"{M['auc']:.2f}")
    k4.metric("% Prioridad Alta",f"{(df.prioridad=='Alta').mean()*100:.1f}%" if 'prioridad' in df.columns else "-")
    k5.metric("Gasto Prom 12m",fmt_clp(df.monetary_total.mean()) if 'monetary_total' in df.columns else "-")
    k6.metric("Uplift %>0",f"{(df.uplift_x>0).mean()*100:.0f}%" if 'uplift_x' in df.columns else "-")
    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("💡 Oportunidad de Activacion")
        pos=df[df.funnel_state_at_t0=='Posibilidad Canje']
        avg_r=df[df.y>=1].revenue_post_12m.mean() if (df.y>=1).any() else 0
        pot=len(pos)*avg_r*0.15
        st.metric("Clientes listos para canjear",f"{len(pos):,}")
        st.info(f"Si activamos al **15%** de estos clientes, generamos **{fmt_clp(pot)}** en revenue incremental.")
    with c2:
        st.subheader("⚠️ Riesgo de Fuga")
        fuga=df[df.funnel_state_at_t0=='Fuga']
        st.metric("Clientes en Fuga",f"{len(fuga):,}")
        st.warning(f"**{len(fuga):,}** clientes dejaron de canjear. Requieren campaña de reactivacion urgente.")
    st.subheader("Tasa de canje por categoria")
    td=df.groupby('tier').apply(lambda x: pd.Series({'Activacion':(x.y==1).mean()*100,'Recurrencia':(x.y==2).mean()*100})).reset_index()
    fig=px.bar(td,x='tier',y=['Activacion','Recurrencia'],barmode='stack',color_discrete_sequence=['#42A5F5','#EF5350'])
    fig.update_layout(yaxis_title="% clientes",xaxis_title="Categoria",legend_title="Tipo canje")
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("Evolucion temporal")
    if 't0_label' in T.columns:
        fig2=px.line(T,x='t0_label',y='tasa_canje',markers=True,title="Tasa de canje mensual")
        fig2.update_layout(yaxis_title="%",xaxis_title="Mes")
        st.plotly_chart(fig2,use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="💰 CLV & Revenue":
    st.title("💰 Valor del Cliente (CLV) & Revenue")
    st.markdown("*Cuanto vale cada segmento de cliente para el negocio*")
    clv_c=df[df.y>=1].clv_estimado.mean() if (df.y>=1).any() else 0
    clv_n=df[df.y==0].clv_estimado.mean() if (df.y==0).any() else 0
    ratio=clv_c/clv_n if clv_n>0 else 0
    c1,c2,c3=st.columns(3)
    c1.metric("CLV Canjeador",fmt_clp(clv_c)); c2.metric("CLV No Canjeador",fmt_clp(clv_n)); c3.metric("Diferencia",f"{ratio:.1f}x")
    st.success(f"Un cliente que canjea vale **{ratio:.1f}x mas**. Cada activacion nueva multiplica el valor del cliente.")
    st.subheader("CLV por categoria y comportamiento")
    ct=df.groupby(['tier','target_label']).clv_estimado.mean().reset_index()
    fig=px.bar(ct,x='tier',y='clv_estimado',color='target_label',barmode='group',color_discrete_map={'No canjea':'#BDBDBD','Activacion':'#42A5F5','Recurrencia':'#EF5350'})
    fig.update_layout(yaxis_title="CLV promedio ($)",xaxis_title="Categoria",legend_title="Comportamiento")
    st.plotly_chart(fig,use_container_width=True)
    if 'quintil_label' in df.columns:
        st.subheader("CLV por quintil de gasto")
        cq=df.groupby('quintil_label').apply(lambda x: pd.Series({'CLV':x.clv_estimado.mean(),'Tasa canje':(x.y>0).mean()*100})).reset_index()
        fig2=px.bar(cq,x='quintil_label',y='CLV',text=cq['Tasa canje'].apply(lambda x: f'{x:.0f}%'),title="CLV por quintil (texto = tasa de canje)")
        fig2.update_layout(yaxis_title="CLV ($)",xaxis_title="Quintil")
        st.plotly_chart(fig2,use_container_width=True)
    # Revenue expected
    st.subheader("📊 Revenue esperado por probabilidad de canje")
    if 'propensity_score' in df.columns and 'estimated_revenue' in df.columns:
        df_tmp=df.copy(); df_tmp['p_bin']=pd.cut(df_tmp.propensity_score,[0,.1,.2,.3,.5,.7,.9,1],labels=['0-10%','10-20%','20-30%','30-50%','50-70%','70-90%','90-100%'])
        rev=df_tmp.groupby('p_bin').apply(lambda x: pd.Series({'N':len(x),'Revenue esperado':x.estimated_revenue.sum(),'Revenue real':x.revenue_post_12m.sum()})).reset_index()
        fig3=px.bar(rev,x='p_bin',y=['Revenue esperado','Revenue real'],barmode='group',title="Revenue por rango de probabilidad de canje")
        fig3.update_layout(yaxis_title="Revenue ($)",xaxis_title="P(canje)")
        st.plotly_chart(fig3,use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="📊 RFM":
    st.title("📊 Analisis RFM")
    st.markdown("*Recency × Frequency × Monetary — segmentacion clasica de CRM*")
    if 'rfm_segment' in df.columns:
        rfm=df.groupby('rfm_segment').apply(lambda x: pd.Series({'Clientes':len(x),'%':len(x)/len(df)*100,'Tasa canje':(x.y>0).mean()*100,'Recency':x.recency_days.mean(),'Frequency':x.frequency_total.mean(),'Monetary':x.monetary_total.mean(),'CLV':x.clv_estimado.mean()})).reset_index().sort_values('CLV',ascending=False)
        st.dataframe(rfm.style.format({'Clientes':'{:,.0f}','%':'{:.1f}%','Tasa canje':'{:.1f}%','Recency':'{:.0f} dias','Frequency':'{:.1f}','Monetary':'${:,.0f}','CLV':'${:,.0f}'}),use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            fig=px.pie(rfm,names='rfm_segment',values='Clientes',hole=.4,title="Distribucion RFM")
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig2=px.scatter(df.sample(min(2000,len(df))),x='recency_days',y='frequency_total',size='monetary_total',color='rfm_segment',title="R × F (tamaño = Monetary)",opacity=0.6)
            fig2.update_layout(xaxis_title="Recency (dias, menor=mejor)",yaxis_title="Frequency (compras)")
            st.plotly_chart(fig2,use_container_width=True)
        st.info("**Champions** = clientes recientes, frecuentes, alto gasto → retener. **Perdidos** = inactivos, baja frecuencia → reactivar o soltar. **En Riesgo** = fueron buenos pero se estan yendo → campaña urgente.")

# ══════════════════════════════════════════════════════════════
elif V=="🔄 Funnel":
    st.title("🔄 Funnel de Canje")
    st.markdown("*El viaje: Inscrito → Participante → Posibilidad Canje → Canjeador → Recurrente (o Fuga)*")
    states=['Inscrito','Participante','Posibilidad Canje','Canjeador','Recurrente','Fuga']
    if 't0_label' in F.columns:
        fig=px.area(F,x='t0_label',y='n',color='funnel_state_at_t0',category_orders={'funnel_state_at_t0':states},color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(yaxis_title="Clientes",xaxis_title="Mes",legend_title="Estado")
        st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        fd=df.funnel_state_at_t0.value_counts()
        fig2=px.pie(values=fd.values,names=fd.index,hole=.4,title="Distribucion actual",color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2,use_container_width=True)
    with c2:
        fr=df.groupby('funnel_state_at_t0').apply(lambda x: (x.y>0).mean()*100).reset_index(name='Tasa').sort_values('Tasa',ascending=False)
        fig3=px.bar(fr,x='funnel_state_at_t0',y='Tasa',color='Tasa',color_continuous_scale='RdYlGn',title="Probabilidad de canje por estado")
        fig3.update_layout(yaxis_title="% que canjea",xaxis_title="Estado")
        st.plotly_chart(fig3,use_container_width=True)
    biggest=df.funnel_state_at_t0.value_counts().idxmax()
    n_biggest=df.funnel_state_at_t0.value_counts().max()
    st.info(f"🔍 **Cuello de botella:** {biggest} ({n_biggest:,} clientes, {n_biggest/len(df)*100:.0f}%). Aqui es donde las campañas deben enfocarse.")

# ══════════════════════════════════════════════════════════════
elif V=="🧩 Segmentos":
    st.title("🧩 Segmentos de Clientes (K-Means)")
    if 'cluster_name' in df.columns:
        cl=df.groupby('cluster_name').apply(lambda x: pd.Series({'N':len(x),'Tasa':(x.y>0).mean()*100,'Prop':x.propensity_score.mean() if 'propensity_score' in x.columns else 0,'CLV':x.clv_estimado.mean(),'%Alta':(x.prioridad=='Alta').mean()*100 if 'prioridad' in x.columns else 0})).reset_index().sort_values('CLV',ascending=False)
        st.dataframe(cl.style.format({'N':'{:,.0f}','Tasa':'{:.1f}%','Prop':'{:.3f}','CLV':'${:,.0f}','%Alta':'{:.1f}%'}),use_container_width=True)
        acciones={'Heavy Users':'🏆 Upgrade tier, experiencia exclusiva','Exploradores':'🔍 Ofertas multi-retailer','Cazadores de Canje':'🎁 Descuento 20% puntos','Digitales':'📱 App-first, push','Dormidos':'⏰ Reactivacion, puntos x2','En Riesgo':'🚨 Alerta puntos por vencer'}
        for c2 in sorted(df.cluster_name.dropna().unique()):
            n=len(df[df.cluster_name==c2])
            st.markdown(f"**{c2}** ({n:,}): {acciones.get(c2,'Personalizar')}")

# ══════════════════════════════════════════════════════════════
elif V=="📈 Modelo":
    st.title("📈 Performance del Modelo Predictivo")
    st.markdown("*XGBoost — predice P(canje) a 12 meses*")
    if 'propensity_score' in df.columns:
        try:
            from sklearn.metrics import roc_auc_score,roc_curve
            yb=(df.y>0).astype(int); pp=df.propensity_score.fillna(0)
            auc=roc_auc_score(yb,pp)
            c1,c2,c3=st.columns(3); c1.metric("AUC",f"{auc:.4f}"); c2.metric("Tasa base",f"{yb.mean()*100:.1f}%"); c3.metric("Clientes",f"{len(df):,}")
            fpr,tpr,_=roc_curve(yb,pp)
            fig=go.Figure(); fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=f'AUC={auc:.3f}',line=dict(width=3,color='steelblue'))); fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(dash='dash',color='gray'),name='Aleatorio'))
            fig.update_layout(title='Curva ROC',xaxis_title='Falsos positivos',yaxis_title='Verdaderos positivos')
            st.plotly_chart(fig,use_container_width=True)
            st.subheader("Lift por decil")
            base=yb.mean(); dt=df.copy(); dt['dec']=pd.qcut(dt.propensity_score.rank(method='first'),10,labels=range(1,11))
            ld=[]; cum=0; tot=yb.sum()
            for d in range(10,0,-1):
                s=dt[dt.dec==d]; r=(s.y>0).sum(); cum+=r
                ld.append({'Decil':d,'Tasa':r/len(s)*100,'Lift':r/len(s)/base,'Captura':cum/tot*100})
            ldf=pd.DataFrame(ld)
            fig2=make_subplots(specs=[[{"secondary_y":True}]])
            fig2.add_trace(go.Bar(x=ldf.Decil,y=ldf.Lift,name='Lift',marker_color='steelblue'),secondary_y=False)
            fig2.add_trace(go.Scatter(x=ldf.Decil,y=ldf.Captura,name='% capturado',mode='lines+markers',marker_color='red'),secondary_y=True)
            fig2.update_layout(title='Eficiencia de targeting (10=top)',xaxis=dict(dtick=1))
            fig2.update_yaxes(title_text='Lift vs base',secondary_y=False); fig2.update_yaxes(title_text='% canjeadores capturados',secondary_y=True)
            st.plotly_chart(fig2,use_container_width=True)
            t30=ldf[ldf.Decil>=8].Captura.max()
            st.success(f"Contactando al **top 30%** capturamos el **{t30:.0f}%** de canjeadores.")
        except Exception as e: st.warning(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
elif V=="⚡ Incrementalidad":
    st.title("⚡ Incrementalidad — Impacto real del canje en ventas")
    st.markdown("*Comparacion: metodo actual (GitLab) vs modelo causal (PSM)*")
    gl=pd.DataFrame([{'Retail':'FALABELLA','GitLab':28.1,'PSM':21.2},{'Retail':'SODIMAC','GitLab':25.0,'PSM':3.9},{'Retail':'TOTTUS','GitLab':37.0,'PSM':23.5},{'Retail':'FCOM','GitLab':19.3,'PSM':28.9}])
    c1,c2,c3=st.columns(3); c1.metric("Lift GitLab (actual)","+35.3%"); c2.metric("Lift PSM (causal)","+16.2%"); c3.metric("Sobreestimacion","2.2x")
    fig=go.Figure(); fig.add_trace(go.Bar(x=gl.Retail,y=gl.GitLab,name='GitLab (actual)',marker_color='gray')); fig.add_trace(go.Bar(x=gl.Retail,y=gl.PSM,name='PSM (causal)',marker_color='steelblue'))
    fig.update_layout(barmode='group',title='Lift gasto % por retail',yaxis_title='Lift %',xaxis_title='Retail')
    st.plotly_chart(fig,use_container_width=True)
    st.info("**GitLab sobreestima 2.2x** porque los canjeadores ya eran clientes mas activos antes de canjear. PSM controla esta autoseleccion.")
    if 'uplift_x' in df.columns:
        st.subheader("Distribucion del efecto individual")
        p1,p99=df.uplift_x.quantile(0.01),df.uplift_x.quantile(0.99)
        df_clip=df[(df.uplift_x>=p1)&(df.uplift_x<=p99)]
        fig2=px.histogram(df_clip,x='uplift_x',nbins=60,color='prioridad' if 'prioridad' in df.columns else None,color_discrete_map={'Alta':'green','Media':'orange','Baja':'gray','No contactar':'red'},title="Uplift por cliente (sin outliers)")
        fig2.update_layout(xaxis_title="Efecto incremental estimado ($)",yaxis_title="Clientes")
        st.plotly_chart(fig2,use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="🎮 Gamificacion":
    st.title("🎮 Gamificacion — Puntos & Breakage")
    st.markdown("*Cuantos puntos se acumulan, cuantos se canjean, cuantos se pierden*")
    c1,c2,c3=st.columns(3)
    brk=df.breakage.mean()*100 if 'breakage' in df.columns else 75
    c1.metric("Breakage promedio",f"{brk:.0f}%")
    pts_vencer=df.exp_points_current_at_t0.sum() if 'exp_points_current_at_t0' in df.columns else 0
    c2.metric("Puntos por vencer este mes",f"{pts_vencer:,.0f}")
    c3.metric("Valor en CLP (~6.5/pto)",fmt_clp(pts_vencer*6.5))
    st.subheader("Breakage por categoria")
    if 'breakage' in df.columns:
        bk=df.groupby('tier').breakage.mean().reset_index(); bk['breakage']*=100
        fig=px.bar(bk,x='tier',y='breakage',color='breakage',color_continuous_scale='RdYlGn_r',title="% de puntos que nunca se canjean")
        fig.update_layout(yaxis_title="Breakage %",xaxis_title="Categoria")
        st.plotly_chart(fig,use_container_width=True)
    st.subheader("💡 Simulador de reduccion de breakage")
    target_brk=st.slider("Si reducimos el breakage a:",min_value=30,max_value=int(brk),value=int(brk-10),step=5)
    delta_brk=(brk-target_brk)/100
    pts_adicionales=df.stock_points_at_t0.sum()*delta_brk if 'stock_points_at_t0' in df.columns else 0
    rev_adicional=pts_adicionales*6.5
    st.metric(f"Revenue adicional si breakage baja de {brk:.0f}% a {target_brk}%",fmt_clp(rev_adicional))

# ══════════════════════════════════════════════════════════════
elif V=="🔮 Prediccion Proximo Mes":
    st.title("🔮 Prediccion para el proximo mes")
    st.markdown("*Basado en las probabilidades del modelo, que esperamos que pase*")
    n_canjean=(df.propensity_score>0.5).sum() if 'propensity_score' in df.columns else 0
    n_nuevos=((df.propensity_score>0.3)&(df.has_redeemed_before_t0==False)).sum() if 'propensity_score' in df.columns else 0
    n_recur=((df.propensity_score>0.5)&(df.has_redeemed_before_t0==True)).sum() if 'propensity_score' in df.columns else 0
    rev_esp=df.estimated_revenue.sum() if 'estimated_revenue' in df.columns else 0
    pts_esp=(df.propensity_score*df.avg_redeem_points.clip(0)).sum() if 'avg_redeem_points' in df.columns else 0
    pts_vencer=df.exp_points_current_at_t0.sum() if 'exp_points_current_at_t0' in df.columns else 0
    c1,c2,c3=st.columns(3)
    c1.metric("🎯 Clientes que canjearan",f"{n_canjean:,}")
    c2.metric("🆕 Canjeadores nuevos",f"{n_nuevos:,}")
    c3.metric("🔁 Recurrentes",f"{n_recur:,}")
    c4,c5,c6=st.columns(3)
    c4.metric("💰 Revenue esperado",fmt_clp(rev_esp))
    c5.metric("🎁 Puntos a canjear",f"{pts_esp:,.0f}")
    c6.metric("⏰ Puntos por vencer",f"{pts_vencer:,.0f}")
    st.subheader("Apertura por categoria")
    pred=df.groupby('tier').apply(lambda x: pd.Series({
        'Canjearan':(x.propensity_score>0.5).sum() if 'propensity_score' in x.columns else 0,
        'Nuevos':((x.propensity_score>0.3)&(x.has_redeemed_before_t0==False)).sum() if 'propensity_score' in x.columns else 0,
        'Revenue esp':x.estimated_revenue.sum() if 'estimated_revenue' in x.columns else 0,
        'Pts por vencer':x.exp_points_current_at_t0.sum() if 'exp_points_current_at_t0' in x.columns else 0,
    })).reset_index()
    st.dataframe(pred.style.format({'Canjearan':'{:,.0f}','Nuevos':'{:,.0f}','Revenue esp':'${:,.0f}','Pts por vencer':'{:,.0f}'}),use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="🎯 Simulador de Impacto":
    st.title("🎯 Simulador — Que pasa si...")
    st.markdown("*Ajusta los parametros y ve el impacto en revenue*")
    c1,c2=st.columns(2)
    with c1:
        pct_activ=st.slider("% de Posibilidad Canje que activamos",0,50,15,5)
        pct_react=st.slider("% de Fuga que reactivamos",0,30,10,5)
    pos_n=len(df[df.funnel_state_at_t0=='Posibilidad Canje'])
    fuga_n=len(df[df.funnel_state_at_t0=='Fuga'])
    avg_rev_canj=df[df.y>=1].revenue_post_12m.mean() if (df.y>=1).any() else 0
    rev_activ=pos_n*(pct_activ/100)*avg_rev_canj
    rev_react=fuga_n*(pct_react/100)*avg_rev_canj*0.7
    with c2:
        st.metric("Clientes activados",f"{int(pos_n*pct_activ/100):,}")
        st.metric("Clientes reactivados",f"{int(fuga_n*pct_react/100):,}")
        st.metric("Revenue incremental activacion",fmt_clp(rev_activ))
        st.metric("Revenue incremental reactivacion",fmt_clp(rev_react))
        st.metric("**TOTAL INCREMENTAL**",fmt_clp(rev_activ+rev_react))
    st.success(f"Con estas campañas generamos **{fmt_clp(rev_activ+rev_react)}** en revenue incremental, activando {int(pos_n*pct_activ/100):,} clientes nuevos y reactivando {int(fuga_n*pct_react/100):,}.")

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
        c5.metric("P(canje)",f"{r.propensity_score:.1%}" if 'propensity_score' in r.index else "-")
        c6.metric("CLV",fmt_clp(r.clv_estimado) if 'clv_estimado' in r.index else "-")
        c7.metric("Gasto 12m",fmt_clp(r.monetary_total) if 'monetary_total' in r.index else "-")
        c8.metric("Puntos",f"{r.stock_points_at_t0:,.0f}" if 'stock_points_at_t0' in r.index else "-")
        if 'rfm_segment' in r.index: st.info(f"**Segmento RFM:** {r.rfm_segment}")
        st.subheader("Recomendacion")
        for k in ['objetivo','accion','canal','timing']:
            v2=str(r.get(k,'N/A'))
            if v2 and v2!='nan' and v2!='N/A': st.markdown(f"**{k.capitalize()}:** {v2}")

# ══════════════════════════════════════════════════════════════
elif V=="📋 Aperturas":
    st.title("📋 Aperturas & Cross-tabs")
    tab1,tab2,tab3,tab4=st.tabs(["Quintil × Categoria","Cluster × Funnel","Tipo × Prioridad","RFM × Categoria"])
    with tab1:
        if 'quintil_label' in df.columns:
            ct=df.groupby(['quintil_label','tier']).apply(lambda x: (x.y>0).mean()*100).unstack(fill_value=0)
            fig=px.imshow(ct.values,x=ct.columns.tolist(),y=ct.index.tolist(),color_continuous_scale='RdYlGn',aspect='auto',title="% que canjea",labels=dict(color="%"))
            st.plotly_chart(fig,use_container_width=True)
    with tab2:
        if 'cluster_name' in df.columns:
            ct2=pd.crosstab(df.cluster_name,df.funnel_state_at_t0)
            fig2=px.imshow(ct2.values,x=ct2.columns.tolist(),y=ct2.index.tolist(),color_continuous_scale='Blues',aspect='auto',title="N clientes")
            st.plotly_chart(fig2,use_container_width=True)
    with tab3:
        if 'tipo_cliente' in df.columns:
            ct3=pd.crosstab(df.tipo_cliente,df.prioridad,normalize='index')*100
            st.dataframe(ct3.style.format('{:.1f}%'),use_container_width=True)
    with tab4:
        if 'rfm_segment' in df.columns:
            ct4=df.groupby(['rfm_segment','tier']).apply(lambda x: (x.y>0).mean()*100).unstack(fill_value=0)
            fig4=px.imshow(ct4.values,x=ct4.columns.tolist(),y=ct4.index.tolist(),color_continuous_scale='RdYlGn',aspect='auto',title="Tasa canje: RFM × Categoria")
            st.plotly_chart(fig4,use_container_width=True)

# ══════════════════════════════════════════════════════════════
elif V=="💾 Exportar":
    st.title("💾 Exportar listas")
    for gc2,gn in [('prioridad','Prioridad'),('cluster_name','Cluster'),('rfm_segment','RFM')]:
        if gc2 not in df.columns: continue
        st.subheader(f"Por {gn}")
        for v2 in sorted(df[gc2].dropna().unique()):
            s=df[df[gc2]==v2]
            cols=[c2 for c2 in ['cust_id','tier','funnel_state_at_t0','cluster_name','rfm_segment','propensity_score','expected_value','prioridad','canal','timing','clv_estimado'] if c2 in s.columns]
            csv=s[cols].to_csv(index=False)
            st.download_button(f"📥 {v2} ({len(s):,})",csv,f"{gn}_{str(v2).lower().replace(' ','_')}.csv")
