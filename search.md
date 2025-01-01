---
layout: default
title: 搜索
---
## Whataya want from me ❤️‍🩹

<!-- Html Elements for Search -->
<div id="search-container" class="search-container" style="margin: 20px 0;">
  <input type="text" id="search-input" placeholder="请输入要检索内容" style="height: 40px; width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.12);">
  <ul id="results-container" class="results-container" style="list-style: none; padding: 0; margin-top: 10px;"></ul>
</div>

<script src="/assets/js/simple-jekyll-search.min.js" type="text/javascript"></script>

<script>
SimpleJekyllSearch({
  searchInput: document.getElementById('search-input'),
  resultsContainer: document.getElementById('results-container'),
  json: '/search.json',
  searchResultTemplate: '<li style="padding: 10px; border-bottom: 1px solid #eee;"><a href="{url}" style="text-decoration: none; color: #009e87; font-weight: bold;">{title}</a> &nbsp;<span style="color: #0043c7;">{subtitle}</span><p style="color: #666;">{content}</p></li>',
  noResultsText: '没有搜索到相关内容',
  fuzzy: false,
  templateMiddleware: function(prop, value, template) {
    if (prop === "subtitle") {
      if (!value) {
        return "";
      }
      if (value.length > 20) {
        return value.substring(0, 20) + "...";
      }
    }
    if (prop === "content") {
      if (!value) {
        return "";
      }
      const searchTerm = document.getElementById('search-input').value;
      const regex = new RegExp(`(${searchTerm})`, 'gi');
      const highlighted = value.replace(regex, '<mark style="background-color: lightgreen">$1</mark>');
      const startIndex = highlighted.toLowerCase().indexOf(searchTerm.toLowerCase());
      if (startIndex > -1) {
        const start = Math.max(0, startIndex - 100);
        const end = Math.min(highlighted.length, startIndex + 50 + searchTerm.length);
        return highlighted.substring(start, end) + (end < highlighted.length ? "..." : "");
      }
      return highlighted.substring(0, 100) + (highlighted.length > 100 ? "..." : "");
    }
    return value;
  }
})
</script>
