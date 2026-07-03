"""
Genera la página web del TFG (HTML autónomo) con la demo interactiva embebida.

Una landing en primera persona ("lo que hice") + el optimizador de Markowitz
interactivo, usando el código y los datos reales del trabajo. Pensada para
subir a GitHub Pages y funcionar también sin internet.
"""

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cvxpy as cp
from src.optimization.covariance import estimar_covarianza
from src.optimization.constraints import construir_restricciones
from src.optimization.markowitz import cartera_minima_varianza

RAIZ = os.path.join(os.path.dirname(__file__), "..")
MESES = 12

ESTIMADORES = [("muestral", "Muestral"), ("ledoit_wolf", "Ledoit-Wolf"), ("pca", "PCA")]

ESTRATEGIAS = [
    ("Markowitz (muestral)", 0.64, 8.34, 12.97, "opt"),
    ("Markowitz (Ledoit-Wolf)", 0.68, 8.86, 12.99, "opt"),
    ("Markowitz (PCA)", 0.69, 9.00, 13.01, "opt"),
    ("Markowitz + Ridge", 0.68, 9.29, 13.72, "opt"),
    ("Markowitz + Lasso", 0.75, 9.65, 12.79, "opt"),
    ("Markowitz + RF", 0.71, 9.84, 13.91, "opt"),
    ("1/N (equiponderada)", 0.76, 10.68, 14.14, "base"),
    ("SPY (mercado)", 0.90, 12.94, 14.42, "base"),
]


def frontera_con_pesos(mu, Sigma, n_puntos=60, max_peso=0.20):
    w_gmv = cartera_minima_varianza(Sigma, True, max_peso)
    mu_gmv = float(mu @ w_gmv)
    mu_max = float(mu.max())
    targets = np.linspace(mu_gmv, mu_max, n_puntos)
    n = len(mu)
    puntos = []
    for target in targets:
        w = cp.Variable(n)
        restr = construir_restricciones(w, True, max_peso)
        restr.append(mu.values @ w >= target)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma.values)), restr)
        prob.solve()
        if prob.status not in ("optimal", "optimal_inaccurate"):
            continue
        wv = np.clip(np.asarray(w.value).ravel(), 0, None)
        r_m = float(mu.values @ wv)
        s_m = float(np.sqrt(wv @ Sigma.values @ wv))
        puntos.append({
            "ret": round(r_m * MESES * 100, 3),
            "vol": round(s_m * np.sqrt(MESES) * 100, 3),
            "sharpe": round((r_m / s_m) * np.sqrt(MESES), 3) if s_m > 0 else 0.0,
            "w": [round(float(x) * 100, 2) for x in wv],
        })
    gmv = int(np.argmin([p["vol"] for p in puntos]))
    tan = int(np.argmax([p["sharpe"] for p in puntos]))
    return puntos, gmv, tan


def main():
    ret = pd.read_csv(os.path.join(RAIZ, "data", "processed", "retornos_mensuales.csv"),
                      index_col=0, parse_dates=True)
    activos = ret.drop(columns="SPY")
    mu = activos.mean()
    tickers = list(activos.columns)

    estimators = {}
    for key, label in ESTIMADORES:
        print(f"Calculando frontera: {label} ...")
        Sigma = estimar_covarianza(activos, metodo=key)
        puntos, gmv, tan = frontera_con_pesos(mu, Sigma)
        estimators[key] = {"label": label, "frontier": puntos, "gmv": gmv, "tangency": tan}

    cumulative = {"dates": [], "series": {}}
    csv_path = os.path.join(RAIZ, "data", "outputs", "rentabilidades_estrategias.csv")
    if os.path.exists(csv_path):
        rents = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        acum = (1 + rents).cumprod()
        cumulative["dates"] = [d.strftime("%Y-%m") for d in acum.index]
        cumulative["series"] = {c: [round(float(v), 4) for v in acum[c]] for c in acum.columns}

    data = {
        "tickers": tickers,
        "estimators": estimators,
        "strategies": [{"name": n, "sharpe": s, "ret": r, "vol": v, "kind": k}
                       for (n, s, r, v, k) in ESTRATEGIAS],
        "cumulative": cumulative,
    }

    html = HTML_TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
    out = os.path.join(os.path.dirname(__file__), "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nPágina generada: {out}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Optimización de carteras · Markowitz + Machine Learning — TFG</title>
<script src="plotly.min.js"></script>
<style>
  :root{
    --bg:#070b16; --card:#131a2c; --line:#27314d; --txt:#e8edf7; --muted:#8b97b3;
    --accent:#7c5cff; --accent2:#22d3ee; --good:#34d399; --warn:#fbbf24;
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    background:#070a17;color:var(--txt);-webkit-font-smoothing:antialiased;line-height:1.6}
  a{color:inherit;text-decoration:none}
  nav,header,section,footer{position:relative;z-index:1}
  /* AURORA animada que reacciona al scroll */
  .aurora{position:fixed;inset:-30%;z-index:-2;filter:blur(85px);will-change:filter;transition:filter .1s linear}
  .blob{position:absolute;border-radius:50%;opacity:.55;mix-blend-mode:screen;will-change:transform}
  .b1{width:48vw;height:48vw;background:#6d28d9;top:0;left:-4%;animation:f1 22s ease-in-out infinite alternate}
  .b2{width:44vw;height:44vw;background:#0ea5e9;top:26%;right:-8%;animation:f2 27s ease-in-out infinite alternate}
  .b3{width:54vw;height:54vw;background:#c026d3;bottom:-6%;left:16%;animation:f3 31s ease-in-out infinite alternate}
  .b4{width:40vw;height:40vw;background:#2563eb;top:52%;left:-10%;animation:f4 25s ease-in-out infinite alternate}
  .b5{width:34vw;height:34vw;background:#14b8a6;top:8%;right:18%;animation:f1 29s ease-in-out infinite alternate}
  @keyframes f1{to{transform:translate(16vw,14vh) scale(1.25)}}
  @keyframes f2{to{transform:translate(-13vw,18vh) scale(1.15)}}
  @keyframes f3{to{transform:translate(12vw,-13vh) scale(1.12)}}
  @keyframes f4{to{transform:translate(17vw,-9vh) scale(1.28)}}
  .veil{position:fixed;inset:0;z-index:-1;background:linear-gradient(180deg,rgba(7,10,23,.5),rgba(7,10,23,.82))}
  /* fondo de fórmulas matemáticas (parallax con el scroll) */
  .formulas{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none;will-change:transform}
  .formulas span{position:absolute;font-family:"Cambria Math","Latin Modern Math",Georgia,serif;font-style:italic;
    color:#9fb0ff;opacity:.07;white-space:nowrap;text-shadow:0 0 34px rgba(124,92,255,.4);
    animation:drift 24s ease-in-out infinite alternate}
  @keyframes drift{from{transform:translateY(0)}to{transform:translateY(-18px)}}
  /* tarjetas de teoría */
  .theory-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-top:26px}
  @media(max-width:740px){.theory-grid{grid-template-columns:1fr}}
  .fcard{background:linear-gradient(180deg,var(--card),#101729);border:1px solid var(--line);border-radius:16px;padding:20px}
  .fcard .eq{font-family:"Cambria Math",Georgia,serif;font-style:italic;font-size:22px;text-align:center;color:#fff;
    padding:16px 10px;background:#0b1020;border:1px solid var(--line);border-radius:12px;letter-spacing:.02em}
  .fcard h4{margin:14px 0 6px;font-size:15px}
  .fcard p{margin:0;color:var(--muted);font-size:14px}
  .wrap{max-width:1180px;margin:0 auto;padding:0 22px}

  /* NAV */
  nav{position:sticky;top:0;z-index:50;backdrop-filter:blur(12px);
    background:rgba(7,11,22,.72);border-bottom:1px solid var(--line)}
  nav .wrap{display:flex;align-items:center;justify-content:space-between;height:58px}
  nav .brand{font-weight:800;letter-spacing:-.01em}
  nav .links{display:flex;gap:22px;font-size:14px;color:var(--muted)}
  nav .links a:hover{color:var(--txt)}
  @media(max-width:740px){nav .links{display:none}}

  /* HERO */
  .hero{position:relative;overflow:hidden;padding:90px 0 70px}
  .kicker{font-size:12px;letter-spacing:.2em;text-transform:uppercase;color:var(--accent2);font-weight:700}
  .hero h1{font-size:clamp(30px,5vw,52px);line-height:1.08;margin:14px 0 16px;font-weight:850;letter-spacing:-.025em;
    background:linear-gradient(92deg,#fff 10%,#bcacff 55%,var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
  .hero p.lead{font-size:clamp(16px,2vw,20px);color:#c4cce0;max-width:760px}
  .hero .who{margin-top:22px;color:var(--muted);font-size:14px}
  .cta{display:inline-flex;gap:10px;margin-top:30px;flex-wrap:wrap}
  .btn{padding:12px 20px;border-radius:12px;font-weight:700;font-size:14px;cursor:pointer;border:1px solid var(--line);transition:.18s}
  .btn.primary{background:linear-gradient(135deg,var(--accent),#5b8cff);border:0;box-shadow:0 8px 26px rgba(124,92,255,.45)}
  .btn.primary:hover{transform:translateY(-2px)}
  .btn.ghost{background:#0e1426;color:var(--txt)}
  .btn.ghost:hover{border-color:var(--accent2)}

  /* SECTIONS */
  section{padding:64px 0;border-top:1px solid rgba(39,49,77,.5)}
  .eyebrow{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);font-weight:700;margin-bottom:10px}
  h2{font-size:clamp(24px,3.4vw,34px);margin:0 0 16px;font-weight:800;letter-spacing:-.02em}
  .narr{max-width:760px;color:#cdd5e8;font-size:17px}
  .narr b{color:#fff}
  .cards{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-top:28px}
  @media(max-width:740px){.cards{grid-template-columns:1fr}}
  .feat{background:linear-gradient(180deg,var(--card),#101729);border:1px solid var(--line);border-radius:16px;padding:20px}
  .feat .ic{width:42px;height:42px;border-radius:11px;display:grid;place-items:center;font-size:20px;margin-bottom:12px;
    background:linear-gradient(135deg,rgba(124,92,255,.25),rgba(34,211,238,.18));border:1px solid #33406a}
  .feat h4{margin:0 0 6px;font-size:16px}
  .feat p{margin:0;color:var(--muted);font-size:14px}

  /* DEMO */
  .demo-grid{display:grid;grid-template-columns:1.55fr 1fr;gap:18px;margin-top:26px}
  @media(max-width:920px){.demo-grid{grid-template-columns:1fr}}
  .card{background:linear-gradient(180deg,var(--card),#101729);border:1px solid var(--line);
    border-radius:18px;padding:18px;box-shadow:0 12px 44px rgba(0,0,0,.35)}
  .card h3{margin:0 0 12px;font-size:13.5px;color:var(--muted);font-weight:600}
  .card h3 i{color:var(--accent2);font-style:italic}
  .seg{display:inline-flex;background:#0e1426;border:1px solid var(--line);border-radius:12px;padding:4px;gap:4px}
  .seg button{border:0;background:transparent;color:var(--muted);padding:8px 14px;border-radius:9px;font-weight:600;font-size:13px;cursor:pointer;transition:.18s}
  .seg button.active{background:linear-gradient(135deg,var(--accent),#5b8cff);color:#fff;box-shadow:0 6px 18px rgba(124,92,255,.4)}
  .slider-row{margin:16px 2px 4px}
  .slider-row label{display:flex;justify-content:space-between;font-size:13px;color:var(--muted);margin-bottom:8px}
  input[type=range]{width:100%;-webkit-appearance:none;height:6px;border-radius:6px;background:linear-gradient(90deg,var(--accent),var(--accent2));outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;background:#fff;cursor:pointer;box-shadow:0 0 0 4px rgba(124,92,255,.35),0 4px 10px rgba(0,0,0,.5)}
  .metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:4px}
  .metric{background:#0e1426;border:1px solid var(--line);border-radius:14px;padding:14px 12px;text-align:center}
  .metric .v{font-size:26px;font-weight:800;letter-spacing:-.02em}
  .metric .l{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:3px}
  .v.accent{color:var(--accent2)} .v.good{color:var(--good)}
  .chips{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
  .chip{font-size:12px;color:var(--muted);background:#0e1426;border:1px solid var(--line);border-radius:999px;padding:5px 11px}
  .chip b{color:var(--txt)}
  .full{grid-column:1/-1}
  .callout{background:linear-gradient(135deg,rgba(124,92,255,.14),rgba(34,211,238,.10));border:1px solid #33406a;border-radius:14px;padding:16px 18px;font-size:15px;line-height:1.6;margin-top:8px}
  .callout b{color:#fff}
  .pulse{animation:pulse 2s infinite}@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
  footer{padding:46px 0;border-top:1px solid var(--line);color:var(--muted);font-size:13px;text-align:center}
  .reveal{opacity:0;transform:translateY(18px);transition:.7s cubic-bezier(.2,.7,.2,1)}
  .reveal.in{opacity:1;transform:none}
</style>
</head>
<body>
<div class="aurora" id="aurora" aria-hidden="true">
  <span class="blob b1"></span><span class="blob b2"></span><span class="blob b3"></span>
  <span class="blob b4"></span><span class="blob b5"></span>
</div>
<div class="veil" aria-hidden="true"></div>
<div class="formulas" aria-hidden="true">
  <span style="top:7%;left:5%;font-size:40px">min&nbsp; w&#7488;&Sigma;w</span>
  <span style="top:15%;right:6%;font-size:30px">&Sigma;&#8345;&#7522;&#8331;&#8321; w&#7522; = 1</span>
  <span style="top:29%;left:10%;font-size:46px">E[R&#8346;] = w&#7488;&mu;</span>
  <span style="top:43%;right:8%;font-size:34px">&sigma;&#8346;&sup2; = w&#7488;&Sigma;w</span>
  <span style="top:35%;left:45%;font-size:25px">&sigma;&#8346;&sup2; = (A&mu;&sup2; &minus; 2B&mu; + C)/&Delta;</span>
  <span style="top:57%;left:6%;font-size:38px">S = (R&#8346; &minus; r_f) / &sigma;&#8346;</span>
  <span style="top:67%;right:11%;font-size:42px">w* = &Sigma;&#8315;&sup1;(&lambda;&#120793; + &gamma;&mu;)</span>
  <span style="top:81%;left:13%;font-size:30px">&part;&#8466;/&part;w = 0</span>
  <span style="top:88%;right:7%;font-size:34px">&rho;&#7522;&#11388; = &sigma;&#7522;&#11388;/(&sigma;&#7522;&sigma;&#11388;)</span>
</div>

<nav><div class="wrap">
  <div class="brand">Markowitz <span style="color:var(--accent2)">+ ML</span></div>
  <div class="links">
    <a href="#problema">El problema</a>
    <a href="#teoria">Teoría</a>
    <a href="#quehice">Qué hice</a>
    <a href="#demo">Demo</a>
    <a href="#resultados">Resultados</a>
    <a href="#conclusion">Conclusión</a>
  </div>
</div></nav>

<header class="hero"><div class="wrap">
  <div class="kicker">Trabajo Fin de Grado · Ingeniería Matemática · CEU San Pablo</div>
  <h1>Optimización de carteras:<br>Markowitz + Machine Learning</h1>
  <p class="lead">Para mi TFG combiné el modelo clásico de Markowitz con técnicas de <b>Machine Learning</b> para responder a una pregunta: <b>¿se puede batir al mercado optimizando carteras?</b> Mi respuesta, honesta y demostrada con datos reales: casi nunca — y aquí explico por qué.</p>
  <div class="who">Ricardo Marín Fernández-Conde · Director: Alfredo Sánchez Alberca</div>
  <div class="cta">
    <a class="btn primary" href="#demo">▶ Probar el optimizador</a>
    <a class="btn ghost" href="#quehice">Ver qué hice</a>
  </div>
</div></header>

<section id="problema"><div class="wrap reveal">
  <div class="eyebrow">El problema</div>
  <h2>Un modelo elegante con un talón de Aquiles</h2>
  <div class="narr">
    <p>El modelo de Markowitz (1952) es matemáticamente precioso: encuentra la cartera que mejor equilibra <b>rentabilidad y riesgo</b>. Pero al implementarlo me topé con su gran debilidad: necesita conocer la <b>rentabilidad futura</b> de cada activo, y eso es casi imposible de estimar.</p>
    <p>Descubrí que <b>pequeños errores en esa estimación arruinan la cartera "óptima"</b>: el optimizador da pesos enormes a los activos cuya rentabilidad ha sido sobreestimada por azar. Ese <b>error de estimación</b> es el hilo conductor de todo mi trabajo.</p>
  </div>
</div></section>

<section id="teoria"><div class="wrap reveal">
  <div class="eyebrow">La teoría</div>
  <h2>El modelo de Markowitz, en breve</h2>
  <div class="narr"><p>La idea es elegir los pesos <b>w</b> de la cartera que <b>minimizan el riesgo</b> para una rentabilidad deseada. Estas son las cuatro ecuaciones que sostienen todo mi trabajo:</p></div>
  <div class="theory-grid">
    <div class="fcard">
      <div class="eq">E[R&#8346;] = w&#7488;&mu; &nbsp;&middot;&nbsp; &sigma;&#8346;&sup2; = w&#7488;&Sigma;w</div>
      <h4>Rentabilidad y riesgo de la cartera</h4>
      <p>La rentabilidad esperada es la media de los activos ponderada por sus pesos; el riesgo depende de la matriz de covarianzas &Sigma;. Ahí vive la <b>diversificación</b>.</p>
    </div>
    <div class="fcard">
      <div class="eq">min&nbsp; w&#7488;&Sigma;w &nbsp; s.a. &nbsp; w&#7488;&mu; = &mu;*, &nbsp; &Sigma;w&#7522; = 1, &nbsp; w &ge; 0</div>
      <h4>El problema de optimización</h4>
      <p>Minimizo la varianza sujeta a una rentabilidad objetivo y a que los pesos sumen 1 (y sean positivos). Es un <b>programa cuadrático convexo</b>: tiene solución única y exacta.</p>
    </div>
    <div class="fcard">
      <div class="eq">&sigma;&#8346;&sup2; = (A&mu;&#8346;&sup2; &minus; 2B&mu;&#8346; + C) / &Delta;</div>
      <h4>La frontera eficiente</h4>
      <p>Resolviendo el problema para cada nivel de rentabilidad obtengo la frontera eficiente: una <b>hipérbola</b> en el plano riesgo-rentabilidad. Su mitad superior son las carteras óptimas (la curva morada de la demo).</p>
    </div>
    <div class="fcard">
      <div class="eq">S = (E[R&#8346;] &minus; r_f) / &sigma;&#8346;</div>
      <h4>La cartera de máximo Sharpe</h4>
      <p>El ratio de Sharpe mide la rentabilidad por unidad de riesgo. La cartera que lo maximiza es la <b>tangente</b> &mdash; la estrella &#11088; que verás en la demo.</p>
    </div>
  </div>
</div></section>

<section id="quehice"><div class="wrap reveal">
  <div class="eyebrow">Qué hice</div>
  <h2>Mi enfoque, paso a paso</h2>
  <div class="narr"><p>Construí desde cero un proyecto reproducible en Python y lo apliqué a <b>30 acciones del S&amp;P 500 (2010–2024)</b>. Estas son las cuatro piezas:</p></div>
  <div class="cards">
    <div class="feat"><div class="ic">∑</div><h4>Formulé la matemática con rigor</h4><p>Planteé Markowitz como un problema cuadrático convexo y lo resolví de forma exacta, analizando sus condiciones de optimalidad (KKT), la dualidad y la solución analítica de la frontera eficiente.</p></div>
    <div class="feat"><div class="ic">Σ</div><h4>Comparé tres formas de medir el riesgo</h4><p>Implementé tres estimadores de la matriz de covarianza: muestral, shrinkage de Ledoit-Wolf y un modelo de factores por componentes principales (PCA).</p></div>
    <div class="feat"><div class="ic">🤖</div><h4>Apliqué Machine Learning</h4><p>Entrené tres modelos (Ridge, Lasso y Random Forest) para predecir las rentabilidades esperadas, con validación temporal estricta para no usar nunca información del futuro.</p></div>
    <div class="feat"><div class="ic">⏳</div><h4>Lo probé con honestidad</h4><p>Evalué todo con backtesting walk-forward sobre 15 años, comparándolo con estrategias simples (1/N y el índice) y midiendo si las diferencias eran estadísticamente significativas.</p></div>
  </div>
</div></section>

<section id="demo"><div class="wrap reveal">
  <div class="eyebrow">Pruébalo tú mismo</div>
  <h2>Mi optimizador, en vivo</h2>
  <div class="narr"><p>Esta demo usa <b>mi código y mis datos reales</b>. Mueve el deslizador para recorrer la frontera eficiente y cambia el estimador de covarianza: las carteras y sus pesos se recalculan al instante.</p></div>

  <div class="demo-grid">
    <div class="card">
      <h3>Frontera eficiente — óptimo <i>teórico</i> sobre los datos históricos completos</h3>
      <div id="frontier" style="height:420px"></div>
      <div class="slider-row">
        <label><span>Rentabilidad objetivo de la cartera</span><span id="targetLbl" class="pulse"></span></label>
        <input id="target" type="range" min="0" max="59" value="30">
      </div>
    </div>
    <div class="card">
      <h3>Estimador de la matriz de covarianza &Sigma;</h3>
      <div class="seg" id="seg">
        <button data-k="muestral" class="active">Muestral</button>
        <button data-k="ledoit_wolf">Ledoit-Wolf</button>
        <button data-k="pca">PCA</button>
      </div>
      <div class="metrics" style="margin-top:16px">
        <div class="metric"><div class="v accent" id="mRet">—</div><div class="l">Rent. anual</div></div>
        <div class="metric"><div class="v" id="mVol">—</div><div class="l">Volatilidad</div></div>
        <div class="metric"><div class="v good" id="mSharpe">—</div><div class="l">Sharpe</div></div>
      </div>
      <div class="chips">
        <span class="chip"><b id="nAct">—</b> activos con peso &gt; 1%</span>
        <span class="chip">Tope por activo: <b>20%</b></span>
        <span class="chip" id="tagPoint"></span>
      </div>
      <h3 style="margin-top:20px">Pesos de la cartera seleccionada</h3>
      <div id="weights" style="height:300px"></div>
    </div>
  </div>
</div></section>

<section id="resultados"><div class="wrap reveal">
  <div class="eyebrow">Mis resultados</div>
  <h2>La parte honesta: nadie gana al mercado</h2>
  <div class="narr"><p>Cuando comparé todas las estrategias <b>fuera de muestra</b>, ninguna cartera optimizada superó al mercado. Es más: al aplicar tests de significancia, comprobé que <b>ninguna diferencia es estadísticamente fiable</b>.</p></div>
  <div class="card full" style="margin-top:22px">
    <h3>Ratio de Sharpe por estrategia — backtesting walk-forward (143 meses fuera de muestra)</h3>
    <div id="bars" style="height:360px"></div>
    <div class="callout">
      🔍 <b>Mi hallazgo clave:</b> la mejor cartera optimizada (Lasso, 0,75) solo <b>iguala</b> a la diversificación ingenua 1/N (0,76), y ninguna bate al mercado (SPY, 0,90). Sometido al test de Memmel y a un bootstrap, <b>ninguna diferencia es significativa</b>.
      <br><br>⚖️ <b>El contraste que más me gusta:</b> arriba, la frontera <i>teórica</i> promete un Sharpe altísimo; aquí, fuera de muestra, se desvanece. <b>Esa brecha es el error de estimación</b>, justo lo que mi trabajo cuantifica.
    </div>
  </div>
  <div class="card full" id="cumCard" style="margin-top:18px">
    <h3>Evolución de 1&euro; invertido (2013–2024)</h3>
    <div id="cum" style="height:380px"></div>
  </div>
</div></section>

<section id="conclusion"><div class="wrap reveal">
  <div class="eyebrow">Mi conclusión</div>
  <h2>Por qué esto sí tiene valor</h2>
  <div class="narr">
    <p>Mi aportación <b>no es un modelo que gane al mercado</b>, sino una <b>demostración rigurosa de por qué es tan difícil conseguirlo</b>. Aprendí que, con datos reales y parámetros estimados, el error de estimación domina cualquier sofisticación del modelo.</p>
    <p>Y demostré algo igual de importante: que <b>la honestidad estadística vale más que un resultado "ganador"</b> que no resistiría un contraste serio. Como líneas futuras propongo medidas de riesgo de <i>downside</i> (CVaR) y un análisis de robustez más amplio.</p>
  </div>
</div></section>

<footer><div class="wrap">
  Ricardo Marín Fernández-Conde · CEU Universidad San Pablo · Ingeniería Matemática<br>
  Demo y resultados generados a partir del pipeline reproducible del TFG.
</div></footer>

<script>
const DATA = /*__DATA__*/;
const PLOT_BG="rgba(0,0,0,0)";
const FONT={color:"#cdd6ee",family:"-apple-system,Segoe UI,Roboto,sans-serif"};
const GRID="#222c44";
let estimator="muestral";
const baseLayout=(o={})=>Object.assign({paper_bgcolor:PLOT_BG,plot_bgcolor:PLOT_BG,font:FONT,
  margin:{l:55,r:18,t:12,b:45},showlegend:false,
  xaxis:{gridcolor:GRID,zerolinecolor:GRID},yaxis:{gridcolor:GRID,zerolinecolor:GRID}},o);

function drawFrontier(){
  const e=DATA.estimators[estimator],fr=e.frontier;
  const sel=fr[+document.getElementById("target").value];
  const gmv=fr[e.gmv],tan=fr[e.tangency];
  const traces=[
    {x:fr.map(p=>p.vol),y:fr.map(p=>p.ret),mode:"lines",line:{color:"#7c5cff",width:4,shape:"spline"},fill:"tozeroy",fillcolor:"rgba(124,92,255,0.06)",hoverinfo:"skip"},
    {x:[gmv.vol],y:[gmv.ret],mode:"markers+text",text:["GMV"],textposition:"bottom center",textfont:{color:"#22d3ee"},marker:{color:"#22d3ee",size:11,line:{color:"#fff",width:1}},hoverinfo:"skip"},
    {x:[tan.vol],y:[tan.ret],mode:"markers+text",text:["Máx. Sharpe"],textposition:"top center",textfont:{color:"#34d399"},marker:{color:"#34d399",size:11,symbol:"star",line:{color:"#fff",width:1}},hoverinfo:"skip"},
    {x:[sel.vol],y:[sel.ret],mode:"markers",marker:{color:"#fff",size:16,line:{color:"#7c5cff",width:3}},hovertemplate:"Vol %{x:.1f}%<br>Rent %{y:.1f}%<extra></extra>"}
  ];
  Plotly.react("frontier",traces,baseLayout({xaxis:{title:"Volatilidad anual (%)",gridcolor:GRID},yaxis:{title:"Rentabilidad anual (%)",gridcolor:GRID}}),{displayModeBar:false,responsive:true});
}
function drawWeights(){
  const e=DATA.estimators[estimator];
  const sel=e.frontier[+document.getElementById("target").value];
  const pairs=DATA.tickers.map((t,i)=>[t,sel.w[i]]).filter(p=>p[1]>1.0).sort((a,b)=>b[1]-a[1]);
  Plotly.react("weights",[{x:pairs.map(p=>p[1]),y:pairs.map(p=>p[0]),type:"bar",orientation:"h",
    marker:{color:pairs.map(p=>p[1]),colorscale:[[0,"#3b3a8f"],[1,"#22d3ee"]],line:{width:0}},
    hovertemplate:"%{y}: %{x:.1f}%<extra></extra>"}],
    baseLayout({margin:{l:60,r:18,t:6,b:30},xaxis:{title:"Peso (%)",gridcolor:GRID},yaxis:{autorange:"reversed",gridcolor:"rgba(0,0,0,0)"}}),
    {displayModeBar:false,responsive:true});
  document.getElementById("nAct").textContent=pairs.length;
}
function updateMetrics(){
  const e=DATA.estimators[estimator],i=+document.getElementById("target").value,sel=e.frontier[i];
  mRet.textContent=sel.ret.toFixed(1)+"%";mVol.textContent=sel.vol.toFixed(1)+"%";mSharpe.textContent=sel.sharpe.toFixed(2);
  targetLbl.textContent=sel.ret.toFixed(1)+"% anual";
  let tag=i===e.gmv?"📍 Mínima varianza (GMV)":(i===e.tangency?"⭐ Máximo Sharpe (tangente)":"");
  tagPoint.textContent=tag;tagPoint.style.display=tag?"inline-block":"none";
}
function refresh(){drawFrontier();drawWeights();updateMetrics();}
document.getElementById("target").addEventListener("input",refresh);
document.querySelectorAll("#seg button").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll("#seg button").forEach(x=>x.classList.remove("active"));
  b.classList.add("active");estimator=b.dataset.k;
  document.getElementById("target").value=DATA.estimators[estimator].tangency;refresh();
}));
function drawBars(){
  const s=DATA.strategies.slice().sort((a,b)=>a.sharpe-b.sharpe);
  Plotly.react("bars",[{x:s.map(x=>x.sharpe),y:s.map(x=>x.name),type:"bar",orientation:"h",
    marker:{color:s.map(x=>x.kind==="base"?"#fbbf24":"#7c5cff"),line:{width:0}},
    text:s.map(x=>x.sharpe.toFixed(2)),textposition:"outside",textfont:{color:"#e8edf7"},
    hovertemplate:"%{y}<br>Sharpe %{x:.2f}<extra></extra>"}],
    baseLayout({margin:{l:185,r:34,t:6,b:40},xaxis:{title:"Ratio de Sharpe (anualizado)",gridcolor:GRID,range:[0,1.05]},yaxis:{gridcolor:"rgba(0,0,0,0)"}}),
    {displayModeBar:false,responsive:true});
}
function drawCum(){
  const c=DATA.cumulative;
  if(!c.dates||!c.dates.length){document.getElementById("cumCard").style.display="none";return;}
  const hi={"SPY (mercado)":"#fff","1/N (equiponderada)":"#fbbf24","Markowitz + Lasso":"#34d399"};
  const traces=Object.keys(c.series).map(name=>({x:c.dates,y:c.series[name],mode:"lines",name,
    line:{color:hi[name]||"#5566aa",width:hi[name]?3:1.2,dash:(name.includes("SPY")||name.includes("1/N"))?"dot":"solid"},
    opacity:hi[name]?1:0.5,hovertemplate:name+": %{y:.2f}€<extra></extra>"}));
  Plotly.react("cum",traces,baseLayout({showlegend:true,legend:{font:{size:10,color:"#aab"},orientation:"h",y:-0.18},
    margin:{l:50,r:20,t:6,b:60},xaxis:{gridcolor:GRID},yaxis:{title:"Valor (€)",gridcolor:GRID}}),
    {displayModeBar:false,responsive:true});
}
// reveal on scroll
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)e.target.classList.add("in")}),{threshold:.12});
document.querySelectorAll(".reveal").forEach(el=>io.observe(el));
// init
document.getElementById("target").value=DATA.estimators[estimator].tangency;
refresh();drawBars();drawCum();
// fondo interactivo: el tono de la aurora rota y las fórmulas se desplazan con el scroll
const _aurora=document.getElementById("aurora"), _formulas=document.querySelector(".formulas");
function onScroll(){
  const y=window.scrollY||0, max=Math.max(1,document.body.scrollHeight-window.innerHeight), p=Math.min(1,y/max);
  _aurora.style.filter="blur(85px) hue-rotate("+(p*160).toFixed(1)+"deg) saturate("+(1+p*0.4).toFixed(2)+")";
  _formulas.style.transform="translateY("+(y*-0.10).toFixed(1)+"px)";
}
window.addEventListener("scroll",onScroll,{passive:true}); onScroll();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
