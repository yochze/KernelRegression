f = File.open('auto-mpg.data_full.txt')

lines = f.readlines

lines.each do |line|
    sl = line.split("\s\s")
    sl.delete_if { |x| x.size == 0}
    # sl.map! { |x| x.strip! }
    puts sl[0] + "," + sl[1] + "," + sl[3] + ","  + sl[4] + ","  + sl[5]
end
