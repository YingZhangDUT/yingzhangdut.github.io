---
layout: default
title: Moments
permalink: /moments/
main_nav: true
order: 3
---

<!--
{% for moment in site.moments %}
  {% include card.html %}
{% endfor %}
-->

<main class="site__content">
  <section class="moment">
    <div class="moment-container">
      <div class="moment-list" itemscope="" itemtype="http://schema.org/Blog">
		  {% for moment in site.moments %}
		  	{% include card.html %} 
		  {% endfor %}
		  {% if forloop.last == false %}<hr>{% endif %}
      </div>
    </div>
  </section>
</main> 
