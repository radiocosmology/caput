{% extends "!autosummary/class.rst" %}

{% block methods %}
{% if methods %}
    .. autosummary::
       :toctree:
    {% for item in methods %}
       {{ name }}.{{ item }}
    {%- endfor %}
    .. HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
      {% for item in all_methods %}
         {{ name }}.{{ item }}
      {%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
    .. autosummary::
       :toctree:
    {% for item in attributes %}
       {{ name }}.{{ item }}
    {%- endfor %}
    .. HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
      {% for item in all_attributes %}
         {{ name }}.{{ item }}
      {%- endfor %}
{% endif %}
{% endblock %}
