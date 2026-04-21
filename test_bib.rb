require 'bibtex'
b = BibTeX.parse("
@article{test,
  author = {Valerio Marsocci and Yuru Jia and Georges Le Bellier and David Kerekes and Liang Zeng and Sebastian Hafner and Sebastian Gerard and Eric Brune and Ritu Yadav and Ali Shibli and Heng Fang and Yifang Ban and Maarten Vergauwen and Nicolas Audebert and Andrea Nascetti}
}")
b['test'].author.each do |a|
  puts "First: '#{a.first}', Last: '#{a.last}'"
end
