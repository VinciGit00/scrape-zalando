{% for cell in nb.cells %}
{% if cell.cell_type == 'markdown' %}
{{ cell.source }}
{% elif cell.cell_type == 'code' %}
```python
{{ cell.source }}
```
{% for output in cell.outputs %}
{% if output.output_type == 'stream' %}
{{ output.text }}
{% elif output.output_type == 'display_data' or output.output_type == 'execute_result' %}
{% if 'text/html' in output.data %}
{{ output.data['text/html'] }}
{% elif 'image/png' in output.data %}
![png](data:image/png;base64,{{ output.data['image/png'] }})
{% elif 'text/plain' in output.data %}
{{ output.data['text/plain'] }}
{% endif %}
{% endif %}
{% endfor %}
{% endif %}
{% endfor %}