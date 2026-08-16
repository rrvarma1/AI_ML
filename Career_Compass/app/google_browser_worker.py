"""Isolated Google Careers browser worker for Windows/Uvicorn compatibility.

Google Careers currently renders result cards using client-side navigation, so job
URLs may not exist as ordinary <a href> elements. This worker therefore extracts
routes from anchors/DOM attributes/rendered HTML and finally resolves cards by
clicking their Learn more controls when necessary.
"""
import asyncio
import html as htmlmod
import json
import re
import sys
from urllib.parse import urljoin

BASE = "https://www.google.com"
ROUTE_RE = re.compile(r"(?:https?://www\.google\.com)?(/about/careers/applications/jobs/results/\d+(?:-[A-Za-z0-9_-]+)?)")
SHORT_RE = re.compile(r"(?:https?://(?:www\.)?google\.com)?(/jobs/results/\d+(?:-[A-Za-z0-9_-]+)?)")


def normalize_url(value: str) -> str | None:
    if not value:
        return None
    value = htmlmod.unescape(str(value)).replace('\\/', '/').replace('\\u002F', '/')
    m = ROUTE_RE.search(value)
    if m:
        return urljoin(BASE, m.group(1))
    m = SHORT_RE.search(value)
    if m:
        # Old careers.google.com style URLs still redirect correctly, but normalize
        # to the current applications route when possible.
        return urljoin(BASE, '/about/careers/applications' + m.group(1))
    return None


def title_from_url(url: str) -> str:
    m = re.search(r'/jobs/results/\d+-([^?#/]+)', url)
    if not m:
        return ''
    return ' '.join(w.capitalize() if w not in {'ai','ml','ux','ui','ii','iii','iv'} else w.upper() for w in m.group(1).split('-'))


def card_text_for_element(el):
    try:
        return el.evaluate("""el => {
          let n = el;
          for (let i=0; i<8 && n; i++, n=n.parentElement) {
            const t=(n.innerText||'').trim();
            if (t.length >= 40 && t.length <= 4500 &&
                (t.includes('Minimum qualifications') || t.includes('Learn more'))) return t;
          }
          return (el.parentElement && el.parentElement.innerText || el.innerText || '').trim();
        }""")
    except Exception:
        return ''


def infer_location(card_text: str) -> str:
    """Best-effort extraction of the visible location from a Google result card."""
    t=(card_text or '').replace('\r','\n')
    # Material icon text is often flattened as: 'corporate_fare Google place Bengaluru, Karnataka, India bar_chart Mid'.
    m=re.search(r'\bplace\s+(.+?)(?=\s+bar_chart\b|\s+(?:Early|Mid|Advanced)\b|\n|$)',t,re.I)
    if m:
        loc=re.sub(r'\s+',' ',m.group(1)).strip(' ;-')
        if 2 <= len(loc) <= 220:return loc
    lines=[re.sub(r'\s+',' ',x).strip() for x in t.splitlines() if x.strip()]
    for line in lines[1:]:
        if ',' in line and len(line) <= 220 and not re.search(r'qualification|experience|degree|apply|learn more',line,re.I):
            # Google result locations normally contain comma-separated city/state/country values.
            return re.sub(r'^(?:Google|YouTube|DeepMind)\s+','',line).strip()
    return ''


def add_row(rows_by_url, href, text='', parent=''):
    href = normalize_url(href)
    if not href:
        return
    if href not in rows_by_url:
        rows_by_url[href] = {
            'href': href,
            'text': (text or '').strip() or title_from_url(href),
            'parent': (parent or text or '').strip(),
            'location': infer_location(parent or text or ''),
        }
    else:
        if text and len(text.strip()) > len(rows_by_url[href].get('text','')):
            rows_by_url[href]['text'] = text.strip()
        if parent and len(parent.strip()) > len(rows_by_url[href].get('parent','')):
            rows_by_url[href]['parent'] = parent.strip()
            rows_by_url[href]['location'] = infer_location(parent)


def extract_rendered(page):
    rows = {}

    # 1) Ordinary anchors (works on some Google deployments/locales).
    try:
        anchors = page.locator('a[href]').evaluate_all("""els => els.map(a => ({
          href:a.href || a.getAttribute('href') || '',
          text:(a.innerText||a.textContent||'').trim(),
          parent:(a.parentElement && a.parentElement.innerText || '').trim()
        }))""")
        for row in anchors:
            add_row(rows, row.get('href',''), row.get('text',''), row.get('parent',''))
    except Exception:
        pass

    # 2) Angular/router/data attributes. Google can make cards clickable without href.
    try:
        attrs = page.locator('*').evaluate_all("""els => {
          const out=[];
          for (const el of els) {
            for (const a of Array.from(el.attributes||[])) {
              const v=a.value||'';
              if (v.includes('jobs/results/') || v.includes('careers/applications/jobs/results/')) {
                out.push({value:v,text:(el.innerText||el.textContent||'').trim(), parent:(el.parentElement && el.parentElement.innerText || '').trim()});
              }
            }
          }
          return out;
        }""")
        for row in attrs:
            add_row(rows, row.get('value',''), row.get('text',''), row.get('parent',''))
    except Exception:
        pass

    # 3) Routes serialized into framework state/scripts or other rendered HTML.
    try:
        rendered = page.content().replace('\\/', '/').replace('\\u002F', '/')
        for m in ROUTE_RE.finditer(rendered):
            add_row(rows, m.group(0))
        for m in SHORT_RE.finditer(rendered):
            add_row(rows, m.group(0))
    except Exception:
        pass

    return rows


def enrich_rows_from_cards(page, rows):
    """Attach rendered-card location/context to URLs recovered from router state."""
    if not rows:
        return rows
    contexts=[]
    try:
        controls=page.get_by_text('Learn more', exact=True)
        for i in range(min(controls.count(), 40)):
            ctx=card_text_for_element(controls.nth(i))
            if ctx:
                contexts.append(ctx)
    except Exception:
        return rows

    def toks(v):
        return set(re.findall(r'[a-z0-9]+',(v or '').lower())) - {'google','youtube','deepmind','the','and','for','with'}

    for row in rows.values():
        if row.get('location'):
            continue
        title=row.get('text') or title_from_url(row.get('href',''))
        tt=toks(title)
        best=None;best_score=0.0
        for ctx in contexts:
            # Only compare the first part of a card so qualifications don't create false matches.
            head=' '.join(ctx.splitlines()[:8])[:900]
            ct=toks(head)
            if not tt or not ct:
                continue
            score=len(tt & ct)/max(1,len(tt))
            if score>best_score:
                best_score=score;best=ctx
        if best is not None and best_score>=0.45:
            row['parent']=best
            row['location']=infer_location(best)
    return rows


def click_fallback(page, rows, search_url):
    """Resolve Google client-side job cards by clicking Learn more controls.

    This is slower, so it is only used when DOM/serialized-route extraction finds no
    job URLs. Each click is followed by reading page.url and navigating back.
    """
    try:
        controls = page.get_by_text('Learn more', exact=True)
        count = min(controls.count(), 20)
    except Exception:
        count = 0
    if count:
        print(f"[worker] click fallback: {count} Learn more controls", file=sys.stderr)

    for idx in range(count):
        try:
            # Re-acquire after every navigation because the DOM is rebuilt.
            control = page.get_by_text('Learn more', exact=True).nth(idx)
            context = card_text_for_element(control)
            before = page.url
            control.scroll_into_view_if_needed(timeout=5000)
            control.click(timeout=8000)
            try:
                page.wait_for_url(re.compile(r'.*/jobs/results/\d+.*'), timeout=10000)
            except Exception:
                page.wait_for_timeout(1200)
            resolved = normalize_url(page.url)
            if resolved:
                first = next((ln.strip() for ln in context.splitlines() if ln.strip()), '')
                add_row(rows, resolved, first, context)
            # Return to results page even if URL did not resolve.
            if page.url != before:
                page.go_back(wait_until='domcontentloaded', timeout=15000)
                try:
                    page.wait_for_load_state('networkidle', timeout=5000)
                except Exception:
                    pass
            elif page.url != search_url:
                page.goto(search_url, wait_until='domcontentloaded', timeout=15000)
        except Exception as exc:
            print(f"[worker] click fallback item {idx}: {type(exc).__name__}: {exc}", file=sys.stderr)
            try:
                if '/jobs/results/' in page.url:
                    page.go_back(wait_until='domcontentloaded', timeout=15000)
            except Exception:
                pass
    return rows


def main():
    if sys.platform == 'win32' and hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(json.dumps({'ok': False, 'error': 'Playwright is not installed. Run: pip install playwright && python -m playwright install chromium'}))
        return 2

    if len(sys.argv) < 2:
        print(json.dumps({'ok': False, 'error': 'Missing Google Careers URL'}))
        return 2

    url = sys.argv[1]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    viewport={'width': 1440, 'height': 1200},
                    locale='en-US',
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36'
                )
                page = context.new_page()
                page.goto(url, wait_until='domcontentloaded', timeout=45000)
                try:
                    page.wait_for_function("() => document.body && /jobs matched|Jobs search results/i.test(document.body.innerText)", timeout=20000)
                except Exception:
                    pass
                try:
                    page.wait_for_load_state('networkidle', timeout=10000)
                except Exception:
                    pass

                rows = extract_rendered(page)
                if not rows:
                    rows = click_fallback(page, rows, url)
                rows = enrich_rows_from_cards(page, rows)

                body_text = ''
                try:
                    body_text = page.locator('body').inner_text(timeout=3000)
                except Exception:
                    pass
                matched = None
                m = re.search(r'(\d[\d,]*)\s+jobs matched', body_text, re.I)
                if m:
                    matched = m.group(1)

                print(json.dumps({
                    'ok': True,
                    'rows': list(rows.values()),
                    'diagnostics': {
                        'resolved_urls': len(rows),
                        'google_jobs_matched_text': matched,
                        'learn_more_count': page.get_by_text('Learn more', exact=True).count() if page else 0,
                    }
                }))
                return 0
            finally:
                browser.close()
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
