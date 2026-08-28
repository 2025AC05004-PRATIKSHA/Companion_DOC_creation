import re
import os
from playwright.sync_api import sync_playwright

def compile_tex_to_pdf(tex_path, pdf_out_path):
    with open(tex_path, 'r', encoding='utf-8') as f:
        tex = f.read()

    # Extract title banner info
    banner_m = re.search(r'\\titlebanner\s*\{([^}]*)\}\s*\{([^}]*)\}\s*\{([^}]*)\}\s*\{([^}]*)\}', tex, re.DOTALL)
    label, title, subtitle, meta = ('', '', '', '')
    if banner_m:
        label, title, subtitle, meta = banner_m.groups()

    body = tex
    if banner_m:
        body = body[banner_m.end():]

    if '\\end{document}' in body:
        body = body[:body.index('\\end{document}')]

    # Convert sections
    body = re.sub(r'\\section\*?\{([^}]*)\}', r'<h2 class="sec">\1</h2>', body)
    body = re.sub(r'\\secsub\{([^}]*)\}', r'<div class="secsub">\1</div>', body)

    # Convert environments
    body = re.sub(r'\\begin\{intuition\}(.*?)\\end\{intuition\}', r'<div class="callout intuition"><span class="pill">☞ The intuition</span><div>\1</div></div>', body, flags=re.DOTALL)
    body = re.sub(r'\\begin\{everyday\}(.*?)\\end\{everyday\}', r'<div class="callout everyday"><span class="pill">★ Everyday picture</span><div>\1</div></div>', body, flags=re.DOTALL)
    body = re.sub(r'\\begin\{watchout\}(.*?)\\end\{watchout\}', r'<div class="callout watchout"><span class="pill">✗ Watch out</span><div>\1</div></div>', body, flags=re.DOTALL)
    body = re.sub(r'\\begin\{keytake\}(.*?)\\end\{keytake\}', r'<div class="callout keytake"><span class="pill">✓ Key takeaway</span><div>\1</div></div>', body, flags=re.DOTALL)

    # Worked environment
    body = re.sub(r'\\begin\{worked\}\{([^}]*)\}(.*?)\\end\{worked\}', r'<div class="callout worked"><span class="pill">✎ Worked example: \1</span><div>\2</div></div>', body, flags=re.DOTALL)

    # Steps list
    body = re.sub(r'\\begin\{steps\}', '<ol class="steps">', body)
    body = re.sub(r'\\end\{steps\}', '</ol>', body)
    body = re.sub(r'\\item\s*', '<li>', body)

    # Figures
    tex_dir = os.path.dirname(os.path.abspath(tex_path))
    def fig_repl(m):
        path = m.group(1)
        cap = m.group(2)
        abs_img = os.path.join(tex_dir, path)
        return f'<div class="fig"><img src="file://{abs_img}"><div class="cap">{cap}</div></div>'
    body = re.sub(r'\\housefig\{([^}]*)\}\{([^}]*)\}', fig_repl, body)

    # Vocab & inline styling
    body = re.sub(r'\\vocab\{([^}]*)\}', r'<span class="vocab">\1</span>', body)
    body = re.sub(r'\\textperiodcentered', '·', body)
    body = re.sub(r'\\textbf\{([^}]*)\}', r'<strong>\1</strong>', body)
    body = re.sub(r'\\emph\{([^}]*)\}', r'<em>\1</em>', body)
    body = re.sub(r'\\\\', '<br>', body)

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-svg.min.js"></script>
<style>
@page {{ size: A4; margin: 16mm; }}
body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.5; color: #1A202C; margin: 0; padding: 0; }}
.banner {{ background-color: #21355E; color: white; padding: 18px; border-radius: 6px; margin-bottom: 20px; }}
.banner .brand {{ font-size: 11pt; font-weight: bold; font-family: sans-serif; }}
.banner .label {{ font-size: 8pt; text-transform: uppercase; letter-spacing: 1px; color: #E2E8F0; margin-top: 8px; }}
.banner h1 {{ font-size: 22pt; margin: 6px 0; line-height: 1.2; color: white; }}
.banner .sub {{ font-size: 11pt; color: #CBD5E1; }}
.banner .meta {{ font-size: 8.5pt; color: #94A3B8; border-top: 1px solid #475569; padding-top: 8px; margin-top: 10px; }}

h2.sec {{ color: #21355E; font-size: 13pt; border-bottom: 1.5px solid #21355E; padding-bottom: 4px; margin-top: 22px; page-break-after: avoid; }}
.secsub {{ font-weight: bold; font-style: italic; color: #2C5AA0; font-size: 10.5pt; margin-bottom: 12px; }}

.callout {{ border-radius: 6px; padding: 12px 16px; margin: 14px 0; position: relative; page-break-inside: avoid; }}
.callout .pill {{ display: inline-block; font-weight: bold; font-size: 8.5pt; color: white; padding: 3px 10px; border-radius: 12px; margin-bottom: 8px; }}

.intuition {{ background-color: #F4F7FC; border: 1px solid #2C5AA0; }}
.intuition .pill {{ background-color: #2C5AA0; }}

.everyday {{ background-color: #FBF1E3; border: 1px solid #8A5A1E; }}
.everyday .pill {{ background-color: #8A5A1E; }}

.worked {{ background-color: #F1F8F3; border: 1px solid #2E7D52; }}
.worked .pill {{ background-color: #2E7D52; }}

.watchout {{ background-color: #FBF1F2; border: 1px solid #B23A48; }}
.watchout .pill {{ background-color: #B23A48; }}

.keytake {{ background-color: #F4F0F9; border: 1px solid #6A4C93; }}
.keytake .pill {{ background-color: #6A4C93; }}

.steps {{ margin: 8px 0; padding-left: 20px; }}
.steps li {{ margin-bottom: 6px; }}

.fig {{ text-align: center; margin: 18px 0; page-break-inside: avoid; }}
.fig img {{ max-width: 100%; height: auto; border-radius: 4px; }}
.fig .cap {{ font-size: 8.5pt; color: #64759C; margin-top: 4px; }}

.vocab {{ color: #21355E; font-weight: bold; }}
</style>
</head>
<body>
<div class="banner">
  <div class="brand">WILP, BITS Pilani</div>
  <div class="label">{label}</div>
  <h1>{title}</h1>
  <div class="sub">{subtitle}</div>
  <div class="meta">{meta}</div>
</div>
{body}
</body>
</html>
'''

    tmp_html = tex_path + '.html'
    with open(tmp_html, 'w', encoding='utf-8') as f:
        f.write(html)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f'file://{tmp_html}')
        page.wait_for_timeout(3000)
        page.pdf(path=pdf_out_path, format='A4', margin={'top': '15mm', 'bottom': '15mm', 'left': '15mm', 'right': '15mm'})
        browser.close()

    if os.path.exists(tmp_html):
        os.remove(tmp_html)

compile_tex_to_pdf('/app/output/DNN-session8/companion.tex', '/app/output/DNN-session8/companion.pdf')
compile_tex_to_pdf('/app/output/DNN-session8a/companion.tex', '/app/output/DNN-session8a/companion.pdf')

print('Generated PDF 8 size:', os.path.getsize('/app/output/DNN-session8/companion.pdf'))
print('Generated PDF 8a size:', os.path.getsize('/app/output/DNN-session8a/companion.pdf'))
