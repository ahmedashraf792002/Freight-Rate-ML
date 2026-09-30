import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
ss=getSampleStyleSheet(); B=ParagraphStyle('b',parent=ss['BodyText'],fontSize=9.5,leading=13)
H=ParagraphStyle('h',parent=ss['Heading2'],textColor=colors.HexColor('#064A56'),spaceBefore=8,spaceAfter=3)
cv=pd.read_csv('outputs/cv_results.csv')
def tbl(data,widths):
    t=Table(data,colWidths=widths); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#064A56')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTSIZE',(0,0),(-1,-1),8.5),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#9DAFB3')),('ALIGN',(1,0),(-1,-1),'CENTER')])); return t
s=[Paragraph('Freight Rate Prediction – Validation &amp; Split Report',ss['Title'])]
s+= [Paragraph('1. Data split and validation approach',H),
Paragraph('<b>Key observation:</b> train_test.csv covers 2025-01-01 to 2025-10-31 (48,000 loads) while validation.csv covers 2025-11-01 to 2025-12-31 (12,000 loads). The final task is therefore a <i>forecast into the future</i>, not an interpolation. A random train/test split would leak same-day and same-lane information and overstate accuracy, so I used <b>expanding-window, forward-chaining validation</b>: each fold trains only on data before the fold and tests on the following period (Jul–Aug, Sep, Oct). Hyper-parameters and feature choices were selected on these folds; the final model is refit on all 48,000 labeled loads.',B),
Paragraph('Fold results (model vs. naive baseline = equipment-median $/mile × distance):',B)]
rows=[['Fold','Train n','Test n','MAE $','MAPE %','MedAPE %','Naive MAE $','Naive MAPE %']]
for _,r in cv.iterrows(): rows.append([r.fold,int(r.n_train),int(r.n_test),f'{r.model_MAE:.1f}',f'{r.model_MAPE:.2f}',f'{r.model_MedAPE:.2f}',f'{r.naive_MAE:.1f}',f'{r.naive_MAPE:.2f}'])
m=cv.mean(numeric_only=True); rows.append(['Mean','','',f'{m.model_MAE:.1f}',f'{m.model_MAPE:.2f}',f'{m.model_MedAPE:.2f}',f'{m.naive_MAE:.1f}',f'{m.naive_MAPE:.2f}'])
s+=[tbl(rows,[1.6*inch,.7*inch,.65*inch,.65*inch,.7*inch,.8*inch,.9*inch,.95*inch]),Spacer(1,4),
Paragraph('RMSE (~$628) is dominated by ~1.5% of loads with extreme rates (up to ~$25k; rate-per-mile residuals beyond ±65%) that are not predictable from the features; hence an L1 objective and MAE/MAPE as the primary metrics.',B),
Paragraph('2. Data-quality findings and handling',H),
Paragraph('• <b>Weight</b>: 292 negative values (magnitudes plausible) treated as sign errors → absolute value; 300 missing → kept as NaN with a missing-indicator. <br/>• <b>market_index</b>: 374 missing; it is a daily level with ±0.025 per-row noise. <br/>• <b>Distance</b>: per-route distances vary (std ≈ 19 mi, consistent with road-distance noise) and are floored at 70 miles, which makes the road/great-circle ratio look extreme on very short lanes; these are valid and were kept. <br/>• <b>Label outliers</b>: ~1.5% of rates deviate &gt;65% from comparable loads; handled with an L1 loss rather than deleting rows. <br/>• No duplicate loads; 64 locations, no spelling variants.',B),
Paragraph('3. Model choice',H),
Paragraph('Rate scales roughly linearly with distance, so I model log($/mile) and multiply back by distance. LightGBM captures distance curvature (short-haul premium), equipment and lane geography (lat/lon) with little tuning; results were insensitive to hyper-parameters (MAE $99.9–100.5). <b>market_index was removed</b>: it improved in-sample fit but, out of time, worsened MAE from ~$100 to ~$145 with a -2% to -6% bias, because its relationship to rate is not stable across seasons. <b>quote_signal</b> was removed since its correlation with the residual flips sign month to month (+ in Jan–Mar, Jun, Sep; – in Apr–May, Jul, Oct).',B),
Paragraph('4. Fixed December prediction chart (produced by score.py)',H),
Image('outputs/scorer_results/candidate_december.png',width=6.8*inch,height=6.8*inch*480/1170*1.0),
Paragraph('<b>Caveat:</b> the December inputs have no market or quote values and the model has no seasonal-trend feature, so only the day-of-week pattern (~$811–$827, ±1%) is expressed. This is deliberate: date-trend features cannot extrapolate beyond the training months and were unstable out of time. The level therefore reflects the Jan–Oct average for this lane; any genuine December seasonality is not modeled.',B)]
SimpleDocTemplate('outputs/Freight_Rate_Report.pdf',pagesize=letter,leftMargin=.7*inch,rightMargin=.7*inch,topMargin=.6*inch,bottomMargin=.6*inch).build(s)
