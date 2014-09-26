#system "gpuoverclock 0"
MAX_SIZE  = ARGV.size <= 0 ? 7168 : ARGV[0].to_i
INCREMENT = ARGV.size <= 1 ? 16   : ARGV[1].to_i
TRIALS    = ARGV.size <= 2 ? 1    : ARGV[2].to_i
LABEL     = ARGV.size <= 3 ? ""   : ARGV[3]

TRIALS.times do
  ["oclblas"].each do |blas|
    next if blas == "oclblas" && LABEL == ""
    next if blas != "oclblas" && LABEL != ""
    system "ln -sf Makefile.in.#{blas} Makefile.in"
    system "make clean all"
    ["dgemm","sgemm"
    #["dgemm","dsymm","dsyrk","dsyr2k","dtrmm",
    # "sgemm","ssymm","ssyrk","ssyr2k","strmm"
    ].each do |routine|
      if routine[0].chr == 'd'
        system "cd #{routine}; ruby run.rb #{MAX_SIZE     } #{INCREMENT} 1 #{LABEL}; cd .."
      else
        system "cd #{routine}; ruby run.rb #{MAX_SIZE+1024} #{INCREMENT} 1 #{LABEL}; cd .."
      end
    end
  end
end
#system "gpuoverclock 1"
