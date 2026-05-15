# Sub-folder outline

## File outlines

### geneig warp kernel (cuda file)

This was an attempt to see what happens when I try and write a kernel with non assigned variables inside of the kernel. I jumped the gun and started writing up the accompanying Python file: geneig warp run. etc... before simply trying to compile the cuda file. 

When trying to compile, you get the sort of expected "variables need to be a constant". Essentially, the compiler is trying to allocate actual memory on the hardware since the user didn't specify that the object is a pointer. It can't do this if it doesn't know how much memory to allocate. This 'static' memory allocation is only possible if the user specifies a constant variable and then the variable should live on the stack att the register level. This is part of the reason the original code runs so fast.

### geneig_..._test

This is a fully functioning file with tests and utilizing the string literal approach and the CuPy raw kernel funcitonality. It works because it uses double curly brackets and the actual string literal can be defined at run time. Accompanying tests show essentially the same performance as the stand-alone hardcoded 3 eigenmode code at the equal number of eigenmodes. The tests also compare to a numpy version of the code that follows the same outer product math as the actual reduction kernel and also tests against the CuPy implimentation, with both showing successes.