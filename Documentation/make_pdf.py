import markdown

# 1. Read the Markdown file
with open("README.md", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Convert to HTML
# We convert the mermaid code block into a div so the mermaid.js library can render it
text = text.replace("```mermaid", '<div class="mermaid">')
text = text.replace("```\n\n---", '</div>\n\n---')

html_body = markdown.markdown(text, extensions=['fenced_code', 'tables'])

# 3. Wrap in a beautiful HTML template with styling for PDF printing
html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Topology-Preserving Road Extraction</title>
    <style>
        body {{
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 850px;
            margin: 0 auto;
            padding: 40px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        h1 {{ font-size: 2.2em; }}
        h2 {{ margin-top: 40px; }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 4px;
            font-family: Consolas, monospace;
            color: #e74c3c;
        }}
        ul li {{ margin-bottom: 10px; }}
        
        /* Print-specific styles for generating the PDF */
        @media print {{
            body {{ padding: 0; max-width: 100%; }}
            .mermaid {{ text-align: center; page-break-inside: avoid; margin-bottom: 40px; }}
        }}
    </style>
</head>
<body>

    {html_body}

    <!-- Mermaid library to render the flowchart -->
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'default', flowchart: {{ defaultRenderer: 'elk' }} }});
    </script>
</body>
</html>
"""

# 4. Save the HTML file
with open("README.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("Created README.html successfully!")
