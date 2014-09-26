ROUTINE = "sgemm"
MAX_SIZE  = ARGV.size <= 0 ? 8192 : ARGV[0].to_i
INCREMENT = ARGV.size <= 1 ? 16   : ARGV[1].to_i
TRIALS    = ARGV.size <= 2 ? 1    : ARGV[2].to_i
LABEL     = ARGV.size <= 3 ? ""   : ARGV[3]

def get_blaslib_name
  open("../Makefile.in", "r") do |r|
    return r.gets.strip.scan(/\/([\w\-\.]+)\z/).to_s
  end
  return "unknown"
end

BLASLIB_NAME = get_blaslib_name()
datdir = "dat_#{ROUTINE}"
Dir.mkdir datdir unless File.exist? datdir

error_check = 0

[0].each do |order|
  [0,1].each do |transa|
    [0,1].each do |transb|
      next unless transa==0 && transb==0
      TRIALS.times do
        outfile = "#{datdir}/#{ROUTINE}_"
        outfile += (order  == 0) ? "C" : "R"
        outfile += (transa == 0) ? "N" : "T"
        outfile += (transb == 0) ? "N" : "T"
        outfile += "_#{`hostname`.strip}_#{BLASLIB_NAME}"
        outfile += "_#{LABEL}" unless LABEL==""
        outfile += ".txt"
        puts outfile
        system "./bench_#{ROUTINE} #{order} #{transa} #{transb} #{MAX_SIZE} #{INCREMENT} #{error_check} | tee -a #{outfile}"
      end
    end
  end
end
