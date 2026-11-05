"""Build Online Resources 1-2 (PDF) and 3 (xlsx) from the aggregate CSVs in data/derived.

usage:  python scripts/build_online_resources.py data/derived <out_dir>

Reads only aggregate tables; no patient-level data is touched. Each resource checks the numbers
it prints against the values stated in the manuscript and stops if they disagree.
"""
import sys
import pandas as pd, numpy as np, os
from scipy.stats import binomtest
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FD=os.environ.get('LIBERATION_FONTS', '/usr/share/fonts/truetype/liberation/')
pdfmetrics.registerFont(TTFont('Serif', FD+'LiberationSerif-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Serif-B', FD+'LiberationSerif-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Serif-I', FD+'LiberationSerif-Italic.ttf'))
from reportlab.lib.fonts import addMapping
addMapping('Serif',0,0,'Serif'); addMapping('Serif',1,0,'Serif-B'); addMapping('Serif',0,1,'Serif-I'); addMapping('Serif',1,1,'Serif-B')

A = E = sys.argv[1]; OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
TITLE='Quantum or just quantised? Angle-encoder saturation confounds ansatz and placement comparisons in hybrid models for 3D CT'
JOURNAL='Quantum Machine Intelligence'
AUTH='Virinchi Puvvada, Maneesh Meka, Dhanalaxmi Dhanpal'
CORR='Corresponding author: Virinchi Puvvada, Department of Data Science and Business Systems, SRM Institute of Science and Technology, Chennai, India; virinchi227@gmail.com'

body=ParagraphStyle('b',fontName='Serif',fontSize=10.5,leading=14,spaceAfter=6)
small=ParagraphStyle('s',parent=body,fontSize=9.5,leading=12.5)
h1=ParagraphStyle('h1',parent=body,fontName='Serif-B',fontSize=13,leading=17,spaceAfter=8)
h2=ParagraphStyle('h2',parent=body,fontName='Serif-B',fontSize=11,leading=14,spaceBefore=8)
cell=ParagraphStyle('c',parent=body,fontSize=9.5,leading=11.5,spaceAfter=0)

def header(n, what):
    return [Paragraph(f'Online Resource {n}', h1),
            Paragraph(f'<b>Article:</b> {TITLE}', small),
            Paragraph(f'<b>Journal:</b> {JOURNAL}', small),
            Paragraph(f'<b>Authors:</b> {AUTH}', small),
            Paragraph(CORR, small), Spacer(1,4*mm), Paragraph(what, h2)]

def table(rows, widths, bold_rows=()):
    data=[[Paragraph(str(x), cell) for x in r] for r in rows]
    t=Table(data, colWidths=[w*mm for w in widths], repeatRows=1)
    st=[('LINEABOVE',(0,0),(-1,0),0.8,colors.black),('LINEBELOW',(0,0),(-1,0),0.5,colors.black),
        ('LINEBELOW',(0,-1),(-1,-1),0.8,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]
    t.setStyle(TableStyle(st)); return t

def mk(path, story):
    SimpleDocTemplate(path, pagesize=A4, leftMargin=22*mm, rightMargin=22*mm, topMargin=20*mm, bottomMargin=20*mm,
                      title=f'{os.path.basename(path)}', author=AUTH).build(story)

def sci(p):
    if p>=0.001: return f'{p:.2g}'
    m,e=f'{p:.1e}'.split('e'); return f'{m} × 10<super>{int(e)}</super>'.replace('-','−')
def f4(x): return f'{x:+.4f}'.replace('-', '−') if x<0 or x>0 else '0.0000'
def pfmt(p): return f'{p:.3f}' if p>=0.001 else f'{p:.1e}'.replace('-0','−').replace('e','×10^').replace('×10^','e')

# ---------- Online Resource 1: noise ablation ----------
nz=pd.read_csv(f'{E}/nisq_ablation.csv')
rows=[['Depolarising probability','AUROC','AUPRC','AUROC retained','AUPRC retained']]
for r in nz.itertuples():
    rows.append([f'{r.p_depol:.2f}',f'{r.auroc:.4f}',f'{r.auprc:.4f}',f'{r.auroc_retained:.4f}',f'{r.auprc_retained:.4f}'])
rng=nz.auroc.max()-nz.auroc.min()
story=header(1,'Depolarising-noise ablation')
story+=[Paragraph('Table S1 AUROC and AUPRC with a depolarising channel of probability p applied after each entangling gate. "Retained" is the ratio to the noiseless value (p = 0). Source file nisq_ablation.csv', small),
        table(rows,[38,26,26,32,32]), Spacer(1,4*mm),
        Paragraph(f'Across the four noise levels AUROC varied by {rng:.4f} (range {nz.auroc.min():.4f}–{nz.auroc.max():.4f}) and AUPRC by {nz.auprc.max()-nz.auprc.min():.4f}; values of retained performance slightly above 1 at p = 0.02 and 0.05 are within this variation and should not be read as noise improving performance.', body),
        Paragraph('<b>Scope.</b> This ablation was computed in the earlier, unconditioned configuration with a principal-component reducer and on a pooled arm (main text, Section 5.4). It therefore characterises neither the conditioned models that carry the main results nor the spatially applied construction, and it is a first-order simulation proxy for device noise only. In the unconditioned configuration the encoder is largely saturated (main text, Section 4.1), so insensitivity to gate noise is consistent with the circuit carrying a near-discrete sign code rather than evidence of robustness of a working quantum feature map. It should be read as preliminary. No results were obtained on quantum hardware.', body)]
mk(f'{OUT}/ESM_1.pdf', story)

# ---------- Online Resource 2: TOST + sign test ----------
eq=pd.read_csv(f'{A}/stats_A_equivalence.csv')
rows=[['Cell','Δ (quantum − classical)','90% interval','p (TOST)','Equivalent at ±0.02']]
for r in eq.itertuples():
    rows.append([r.cell, f4(r.delta), f'[{f4(r.lo90)}, {f4(r.hi90)}]', f'{r.p_tost:.4f}', 'yes' if r.equivalent else 'no'])
neq=int(eq.equivalent.sum())
sc=pd.read_csv(f'{E}/stats_C_sign_test.csv').iloc[0]
c=pd.read_csv(f'{A}/stats_C_india_label_budget.csv')
c['backbone']=c.cell.str.split().str[0]
c['endpoint']=np.where(c.cell.str.contains('hemorrhag'),'hemorrhage','any finding')
c['k']=c.cell.str.extract(r'k=(\d+)').astype(int)
# self-checks against the manuscript and against the stored sign-test file
assert len(eq)==12 and neq==9
assert len(c)==36 and int((c.delta<0).sum())==int(sc.n_favour_classical)==33
p_all=binomtest(33,36,0.5).pvalue; assert abs(p_all-sc.binomial_p_two_sided)<1e-12
small_k=c[c.k<=50]; ns=int((small_k.delta<0).sum()); p_small=binomtest(ns,len(small_k),0.5).pvalue
assert ns==10 and len(small_k)==12 and abs(p_small-0.0386)<1e-3
srows=[['Subset','Cells','Favour classical','Favour quantum','Two-sided binomial p']]
def add(label, sub):
    n=len(sub); m=int((sub.delta<0).sum()); srows.append([label,n,m,n-m,sci(binomtest(m,n,0.5).pvalue)])
add('All cells',c)
for e in ['any finding','hemorrhage']: add(f'Endpoint: {e}', c[c.endpoint==e])
for b in ['B1','B2','B3']: add(f'Backbone: {b}', c[c.backbone==b])
add('k ≤ 50 (two smallest budgets)', small_k)
for k in sorted(c.k.unique()): add(f'k = {k}', c[c.k==k])
pos=c[c.delta>0].sort_values('k')
story=header(2,'Equivalence tests (Family A) and cell-level sign test (Family C)')
story+=[Paragraph('Table S2 Two one-sided tests (TOST) of equivalence between the quantum arm and its parameter-matched classical control on the development cohort, all arms conditioned, against a pre-specified margin of ±0.02 AUROC units. A cell is equivalent when its 90% interval lies entirely inside the margin. Five paired seeds per cell. Source file stats_A_equivalence.csv', small),
        table(rows,[26,36,44,22,30]), Spacer(1,3*mm),
        Paragraph(f'{neq} of the 12 comparisons are equivalent at this margin. The three that are not (B2 Q2, B2 Q4, B3 Q2) all have point estimates favouring the classical arm; for those cells the data are consistent with no difference but do not establish equivalence. No cell favours the quantum arm by more than {eq.delta.max():.4f}.', body),
        Spacer(1,3*mm),
        Paragraph('Table S3 Binomial sign test on the 36 cell-level point estimates of Family C (quantum minus parameter-matched classical on the external cohort, trained from scratch, 50 draws per cell). Under the null each cell is equally likely to favour either arm. Because cells share data and are not independent, these p-values are descriptive. Source files stats_C_sign_test.csv and stats_C_india_label_budget.csv', small),
        table(srows,[56,16,28,28,34]), Spacer(1,3*mm),
        Paragraph('The three cells with positive point estimates are all hemorrhage cells on B3 (' + '; '.join(f'k = {r.k}: {f4(r.delta)}, q = {r.q:.3f}' for r in pos.itertuples()) + '); none is significant before or after correction.', body)]
assert set(pos.backbone)=={'B3'} and set(pos.endpoint)=={'hemorrhage'} and (pos.q>0.05).all()
mk(f'{OUT}/ESM_2.pdf', story)

# ---------- Online Resource 3: workbook ----------
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
sheets=[ # (sheet, file, used in)
 ('T1_cohorts', f'{A}/table1_cohorts.csv', 'Table 1'),
 ('T2_conditioning', f'{A}/table1_conditioning.csv', 'Table 2, Figs 1-2'),
 ('placement_arms', f'{A}/table2_placement_and_arms.csv', 'Fig. 3b, Section 4.3'),
 ('T3_famB_placement', f'{A}/stats_B_placement.csv', 'Table 3, Fig. 3b'),
 ('T4_famA_q_vs_c', f'{A}/stats_A_quantum_vs_classical.csv', 'Table 4, Fig. 3a'),
 ('S2_famA_TOST', f'{A}/stats_A_equivalence.csv', 'Online Resource 2, Section 4.4'),
 ('label_efficiency', f'{A}/table3_label_efficiency.csv', 'Figs 4-5'),
 ('T6_famC_label_budget', f'{A}/stats_C_india_label_budget.csv', 'Table 6, Figs 4-5, Section 4.7'),
 ('S3_famC_sign_test', f'{E}/stats_C_sign_test.csv', 'Online Resource 2, Section 4.7'),
 ('transfer', f'{A}/table4_transfer.csv', 'Fig. 6, Section 4.8'),
 ('T7_sweep_lr_wd', f'{E}/sweep_lr_wd.csv', 'Table 7, Section 4.9'),
 ('T8_sweep_qubits', f'{E}/sweep_qubits.csv', 'Table 8, Section 4.9'),
 ('S1_noise_ablation', f'{E}/nisq_ablation.csv', 'Online Resource 1, Section 5.4'),
]
DESC={
 'backbone':'Frozen 3D CT encoder: B1_swinunetr_suprem (B1), B2 and B3 as defined in Section 3.2',
 'family':'Ansatz family (Q1_sel, Q2, Q4) or spatial for the spatially applied construction',
 'placement':'Where the circuit is attached: pooled (after global pooling) or spatial (applied per spatial location)',
 'encoder_norm':'True if the angle encoder is conditioned (normalised before the bounded activation); False if unconditioned',
 'arm':'quantum = variational circuit adapter; classical = parameter-matched classical adapter',
 'auroc_mean':'Mean area under the ROC curve over seeds or draws','auroc_sd':'Standard deviation of AUROC over seeds or draws',
 'pretanh_median':'Median absolute pre-activation |z| entering the bounding activation, after training (quantum arm diagnostic)',
 'pretanh':'Median absolute pre-activation |z| entering the bounding activation, after training',
 'n_seeds':'Number of training seeds','n_draws':'Number of independent patient-grouped label draws',
 'target':'Endpoint: hemorrhage or abnormal_any (any reported finding)','k':'Number of locally labelled training studies (label budget)',
 'mode':'Transfer mode: adaptation depth (e.g. readout) or scratch (local labels only)',
 'cell':'Comparison cell label: backbone, endpoint (abnormal_ = any finding, hemorrhag = hemorrhage) and budget where applicable',
 'delta':'Paired difference, quantum minus classical (or spatial minus pooled for Family B), in AUROC units',
 'lo':'Lower 95% t-interval bound of delta','hi':'Upper 95% t-interval bound of delta',
 'lo90':'Lower 90% interval bound of delta (TOST)','hi90':'Upper 90% interval bound of delta (TOST)',
 'p':'Nominal two-sided paired t-test p-value','q':'Benjamini-Hochberg q-value within the comparison family',
 'n':'Number of paired observations (seeds or draws)','sig_raw':'p < 0.05','sig_bh':'q < 0.05',
 'p_tost':'p-value of the two one-sided tests against a ±0.02 margin','equivalent':'90% interval lies inside ±0.02',
 'n_cells':'Number of Family C cells','n_favour_classical':'Cells with delta < 0','n_favour_quantum':'Cells with delta > 0',
 'binomial_p_two_sided':'Two-sided exact binomial p against 0.5',
 'key':'Run identifier (configuration string; contains no patient information)','lr':'AdamW learning rate','wd':'AdamW weight decay',
 'seed':'Training seed','minutes':'Wall-clock training time (minutes)','ap':'Average precision','prevalence':'Positive fraction of the evaluation set',
 'cal_brier':'Brier score','cal_calibration_slope':'Logistic calibration slope','cal_calibration_intercept':'Logistic calibration intercept',
 'cal_ece':'Expected calibration error','nq':'Number of qubits (as text key)','n_qubits':'Number of qubits',
 'sat_frac':'Fraction of pre-activations in the flat region of the bounding activation (|z| above the saturation threshold)',
 'codes_seen':'Number of distinct sign patterns realised on the evaluation set','codes_possible':'2^n possible sign patterns for n qubits',
 'codes_effective':'Exponential of the Shannon entropy of the sign-pattern distribution','angle_within_code_ratio':'Angle variance within sign patterns as a fraction of total angle variance',
 'auroc_sign_bits':'AUROC of a classifier given only the sign pattern','auroc_code_lut':'AUROC of a lookup table mapping each sign pattern to its training-set positive rate',
 'p_depol':'Depolarising probability applied after each entangling gate','auprc':'Area under the precision-recall curve',
 'auroc':'Area under the ROC curve',
 'Characteristic':'Row label of Table 1','Development cohort (RSNA-ICH)':'Aggregate value for the 3,889 analysed RSNA series (n/r: not recorded in the public release)','External cohort (Nellore)':'Aggregate value for the 1,200 analysed external series; age and sex counted once per patient','auroc_retained':'AUROC divided by the noiseless AUROC','auprc_retained':'AUPRC divided by the noiseless AUPRC',
}
wb=Workbook(); ws=wb.active; ws.title='README'
lines=[TITLE, JOURNAL, AUTH, CORR, '', 'Online Resource 3. Derived, aggregate tables behind the figures and tables of the article.',
 'Every sheet is an unmodified copy of one source CSV (values at full precision). No sheet contains patient-level data, identifiers, images or embeddings.',
 'The external-cohort patient-level data and derived features cannot be released (see Data availability).','',
 'Sheet','Source file','Used in']
for l in lines: ws.append([l])
ws['A1'].font=Font(bold=True,size=12)
for s,f,u in sheets: ws.append([s, os.path.basename(f), u])
ws.append(['T5_transfer_matrix','table5_transfer_matrix.csv','Table 5 (point estimates and patient-clustered bootstrap 95% intervals; written by nb5 cells C2-C3)'])
ws.append(['columns','—','Description of every column'])
for r in ws.iter_rows(min_row=10, max_row=10):
    for c_ in r: c_.font=Font(bold=True)
ws.column_dimensions['A'].width=26; ws.column_dimensions['B'].width=48; ws.column_dimensions['C'].width=60
allcols=[]
for s,f,u in sheets:
    df=pd.read_csv(f); w=wb.create_sheet(s); w.append(list(df.columns))
    for c_ in w[1]: c_.font=Font(bold=True)
    for row in df.itertuples(index=False): w.append([None if (isinstance(v,float) and np.isnan(v)) else (v.item() if hasattr(v,'item') else v) for v in row])
    for col in df.columns:
        if col not in allcols: allcols.append(col)
    w.freeze_panes='A2'
w=wb.create_sheet('T5_transfer_matrix')
_t5=pd.read_csv(f'{A}/table5_transfer_matrix.csv'); w.append(list(_t5.columns))
for row in _t5.itertuples(index=False): w.append(list(row))
for c_ in w[1]: c_.font=Font(bold=True)
for col in _t5.columns:
    if col not in allcols: allcols.append(col)
w=wb.create_sheet('columns'); w.append(['column','description'])
for c_ in w[1]: c_.font=Font(bold=True)
missing=[]
for col in allcols:
    d=DESC.get(col,{'trained_on':'Training cohort','tested_on':'Evaluation cohort','lo95':'Lower 95% patient-clustered bootstrap bound','hi95':'Upper 95% patient-clustered bootstrap bound'}.get(col))
    if d is None: missing.append(col)
    w.append([col,d])
w.column_dimensions['A'].width=28; w.column_dimensions['B'].width=110
assert not missing, missing
wb.save(f'{OUT}/ESM_3.xlsx')
print('ok', neq, p_all, ns, p_small); print(pd.DataFrame(srows[1:],columns=srows[0]).to_string())
