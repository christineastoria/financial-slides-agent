---
name: langchain-brand-slides
description: Generate HTML slide decks styled to the LangChain brand design system in light mode. Use when building financial presentations, reports, or any multi-slide HTML output.
---

# LangChain Brand Slides — Light Mode

Generate self-contained HTML slide decks using the LangChain design system adapted for light backgrounds.

## Slide HTML Structure

Every slide deck is a single HTML string. Each slide is a `<section class="slide">`. The HTML must be fully self-contained with an inline `<style>` block.

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --lc-surface: #F2FAFF;
  --lc-card-alt: #E5F4FF;
  --lc-dark-text: #030710;
  --lc-muted: #6B8299;
  --lc-blue: #7FC8FF;
  --lc-blue-hover: #99D4FF;
  --lc-blue-bg: #E5F4FF;
  --lc-border: #B8DFFF;
  --lc-lime: #4ade80;
  --lc-rose: #f87171;
  --lc-white: #FFFFFF;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

.slide {
  width: 1280px;
  height: 720px;
  background: var(--lc-surface);
  font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
  padding: 48px;
  position: relative;
  overflow: hidden;
  page-break-after: always;
}

.slide.title-slide {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}

h1 {
  font-size: 48px;
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: -0.03em;
  color: var(--lc-dark-text);
  margin-bottom: 16px;
}

h2 {
  font-size: 32px;
  font-weight: 700;
  line-height: 1.25;
  letter-spacing: -0.02em;
  color: var(--lc-dark-text);
  margin-bottom: 24px;
}

h3 {
  font-size: 22px;
  font-weight: 700;
  line-height: 1.3;
  letter-spacing: -0.01em;
  color: var(--lc-dark-text);
}

.subtitle {
  font-size: 18px;
  font-weight: 400;
  color: var(--lc-muted);
  line-height: 1.65;
}

.overline {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--lc-blue);
  margin-bottom: 12px;
}

.metric-card {
  background: var(--lc-white);
  border: 1px solid var(--lc-border);
  border-radius: 16px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.metric-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--lc-muted);
}

.metric-value {
  font-size: 36px;
  font-weight: 800;
  color: var(--lc-dark-text);
  letter-spacing: -0.02em;
}

.metric-trend {
  font-size: 14px;
  font-weight: 600;
}

.metric-trend.up { color: var(--lc-lime); }
.metric-trend.down { color: var(--lc-rose); }
.metric-trend.flat { color: var(--lc-muted); }

.metrics-grid {
  display: grid;
  gap: 20px;
  margin-top: 24px;
}

.metrics-grid.cols-2 { grid-template-columns: repeat(2, 1fr); }
.metrics-grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
.metrics-grid.cols-4 { grid-template-columns: repeat(4, 1fr); }

.content-card {
  background: var(--lc-card-alt);
  border: 1px solid var(--lc-border);
  border-radius: 16px;
  padding: 32px;
}

.slide-number {
  position: absolute;
  bottom: 24px;
  right: 48px;
  font-size: 12px;
  font-weight: 500;
  color: var(--lc-muted);
}

.tag {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--lc-blue);
  background: var(--lc-blue-bg);
  padding: 4px 12px;
  border-radius: 100px;
}
</style>
</head>
<body>
<!-- SLIDES GO HERE -->
</body>
</html>
```

## Slide Types

### Title Slide
```html
<section class="slide title-slide">
  <div class="overline">QUARTERLY REPORT</div>
  <h1>Q4 2024 Business Review</h1>
  <p class="subtitle">Key financial metrics and growth highlights</p>
  <span class="slide-number">1</span>
</section>
```

### Metrics Slide (3 columns)
```html
<section class="slide">
  <div class="overline">REVENUE METRICS</div>
  <h2>Financial Performance</h2>
  <div class="metrics-grid cols-3">
    <div class="metric-card">
      <span class="metric-label">Revenue</span>
      <span class="metric-value">$3.4M</span>
      <span class="metric-trend up">+11.3% vs Q3</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Gross Profit</span>
      <span class="metric-value">$1.6M</span>
      <span class="metric-trend up">+18.5% vs Q3</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Net Income</span>
      <span class="metric-value">$850K</span>
      <span class="metric-trend up">+30.8% vs Q3</span>
    </div>
  </div>
  <span class="slide-number">2</span>
</section>
```

## Rules

1. Always produce a complete self-contained HTML document with the full style block
2. Each slide is a section class slide at exactly 1280x720px
3. Create 4-6 slides: title slide + 3-5 dense content slides. More slides = more thorough.
4. Format currency as $3.4M not $3400000
5. Format percentages with one decimal: 24.6%
6. Use .up (green) for positive trends, .down (red) for negative, .flat (gray) for neutral
7. Prefer .cols-4 grids to pack maximum data per slide. Use .cols-3 only when 4 doesn't divide evenly.
8. Always include slide numbers
9. Never use external images or resources beyond Google Fonts
10. After generating, call the generate_slides tool with the full HTML string
