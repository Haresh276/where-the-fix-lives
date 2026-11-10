import markdown, subprocess, os
os.chdir('/root/wfl/paper')
md=open('paper.md').read()
body=markdown.markdown(md,extensions=['tables']).replace('<p><img','<figure><img').replace('" /></p>','" /></figure>')
body=body.replace('src="fig4_gemma.png"','src="fig4_gemma.png" style="width:78%"')
body=body.replace('src="fig2_per_agent.png"','src="fig2_per_agent.png" style="width:82%"')
body=body.replace('src="fig3_triage_gap.png"','src="fig3_triage_gap.png" style="width:92%"')
css='''@page { size: A4; margin: 18mm 17mm; }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 10.2pt; line-height: 1.42; color:#111; }
h1 { font-size: 17pt; line-height:1.2; margin: 0 0 4pt; } h2 { font-size: 12pt; margin: 14pt 0 4pt; } h3 { font-size: 10.6pt; margin: 10pt 0 3pt; }
h2,h3 { break-after: avoid; page-break-after: avoid; }
p { margin: 0 0 6pt; text-align: justify; } ul,ol { margin: 0 0 6pt 16pt; padding:0 } li { margin-bottom: 2pt; }
figure { margin: 8pt 0 2pt; text-align:center; break-inside: avoid; } img { max-width: 100%; }
table { border-collapse: collapse; font-size: 8.8pt; margin: 4pt auto 8pt; break-inside: avoid; }
th,td { border-bottom: 0.6pt solid #bbb; padding: 2.5pt 6pt; text-align:left; } th { border-bottom: 1pt solid #333; }
code { font-size: 8.8pt; }'''
open('paper.html','w').write(f'<!doctype html><html><head><meta charset="utf-8"><title>Where the Fix Lives</title><style>{css}</style></head><body>{body}</body></html>')
subprocess.run(['node','-e',"const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();await p.goto('file:///root/wfl/paper/paper.html');await p.pdf({path:'/root/wfl/paper/Where_the_Fix_Lives_HareshG.pdf',format:'A4',printBackground:true,margin:{top:'18mm',bottom:'18mm',left:'17mm',right:'17mm'}});await b.close();})()"],check=True)
print('pdf ok')
