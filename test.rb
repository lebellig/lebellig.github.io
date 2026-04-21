require 'liquid'
template = Liquid::Template.parse("
  {%- capture author_list -%}
    {%- for author in authors -%}
      {%- if forloop.first == false -%}, {% endif -%}
      {{ author.first | strip }} {{ author.last | strip }}
    {%- endfor -%}
  {%- endcapture -%}
  RESULT:{{ author_list | strip }},
")
puts template.render('authors' => [{'first' => ' Georges Le ', 'last' => 'Bellier '}, {'first' => 'Nicolas', 'last' => 'Audebert'}])
